import streamlit as st
import pandas as pd
import random
import time
import io
import json
import fitz  # PyMuPDF
from gtts import gTTS
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
from streamlit_autorefresh import st_autorefresh

# ============================================================
# 기본 설정
# ============================================================
st.set_page_config(page_title="영어 단어 받아쓰기 도우미", page_icon="✏️", layout="centered")

QUESTION_SECONDS = 20  # 문제당 제한 시간(초)
GEMINI_MODEL = "gemini-2.0-flash"

BIG_CSS = """
<style>
html, body, [class*="css"]  { font-size: 19px; }
h1 { font-size: 2.3rem !important; }
h2 { font-size: 1.8rem !important; }
h3 { font-size: 1.5rem !important; }
.stButton>button {
    font-size: 1.3rem;
    padding: 0.7em 1.4em;
    border-radius: 14px;
    font-weight: 700;
}
div[data-testid="stMetricValue"] {
    font-size: 3rem;
}
.big-word {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    padding: 0.6em;
    border-radius: 20px;
    background-color: #FFF7E0;
    margin: 0.5em 0;
}
</style>
"""
st.markdown(BIG_CSS, unsafe_allow_html=True)


# ============================================================
# 세션 상태 초기화
# ============================================================
def init_state():
    defaults = {
        "page": "register",              # register -> setup -> test -> result
        "word_bank": pd.DataFrame(columns=["번호", "영어단어", "한글뜻"]),
        "test_questions": [],            # [{id, word, meaning, mode}]
        "current_q": 0,
        "question_start_time": None,
        "played_q": -1,
        "audio_cache": {},                # (text, lang) -> bytes
        "api_key": "",
        "gemini_model": None,             # 자동 감지된 사용 가능 모델명 캐시
        "test_config": {
            "show_answer_box": False,
            "time_limit_seconds": QUESTION_SECONDS,  # None이면 시간 제한 없음
        },
        "user_answers": {},               # 문제 인덱스 -> 입력한 답
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ============================================================
# 공통 유틸
# ============================================================
def get_api_key():
    if st.session_state.api_key:
        return st.session_state.api_key.strip()
    try:
        return st.secrets["GEMINI_API_KEY"].strip()
    except Exception:
        return ""


def get_audio_bytes(text, lang="en"):
    key = (text, lang)
    if key in st.session_state.audio_cache:
        return st.session_state.audio_cache[key]
    tts = gTTS(text=text, lang=lang)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    data = buf.read()
    st.session_state.audio_cache[key] = data
    return data


def _candidate_gemini_models(client):
    """계정에서 실제로 쓸 수 있는 텍스트+이미지 생성 모델 목록을 우선순위대로 반환한다.
    모델마다 무료 할당량이 따로 배정되므로, 하나가 막혀도(404/429) 다른 걸 시도하기 위함."""
    candidates = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            candidates.append(m.name)
    if not candidates:
        raise RuntimeError("사용 가능한 Gemini 모델을 찾지 못했습니다. API 키 권한을 확인해주세요.")

    def rank(name):
        n = name.lower()
        if "flash" in n and "lite" not in n and "preview" not in n and "exp" not in n:
            return 0
        if "flash" in n:
            return 1
        return 2

    ordered = sorted(set(candidates), key=rank)
    cached = st.session_state.get("gemini_model")
    if cached and cached in ordered:
        ordered.remove(cached)
        ordered.insert(0, cached)
    return ordered


def extract_words_from_image(image_bytes, api_key, mime_type="image/jpeg"):
    client = genai.Client(api_key=api_key)
    prompt = (
        "다음 이미지는 초등학생용 영어 단어장을 촬영한 사진입니다.\n"
        "1) 이미지 속에 있는 모든 영어 단어를 순서대로 정확히 추출하세요.\n"
        "2) 각 영어 단어에 대해 초등학교 저학년 학생이 이해하기 쉬운 "
        "쉽고 간단한 한글 뜻을 만들어 주세요.\n"
        "3) 결과는 반드시 JSON 배열로만 응답하세요. 다른 설명은 절대 넣지 마세요.\n"
        '형식 예시: [{"word": "apple", "meaning": "사과"}, {"word": "book", "meaning": "책"}]'
    )
    contents = [prompt, genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]

    cached_model = st.session_state.get("gemini_model")
    response = None
    last_error = None

    # 1) 이전에 성공했던 모델(또는 기본 모델)을 먼저 시도해 API 호출을 아낀다.
    for model_name in [cached_model or GEMINI_MODEL]:
        try:
            response = client.models.generate_content(model=model_name, contents=contents)
            st.session_state.gemini_model = model_name
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            last_error = e

    # 2) 실패했다면(모델 없음/할당량 소진 등) 계정에서 쓸 수 있는 다른 모델들을 순서대로 시도한다.
    if response is None:
        for model_name in _candidate_gemini_models(client):
            try:
                response = client.models.generate_content(model=model_name, contents=contents)
                st.session_state.gemini_model = model_name
                break
            except (genai_errors.ClientError, genai_errors.ServerError) as e:
                last_error = e
                continue

    if response is None:
        raise last_error or RuntimeError("Gemini 요청에 실패했습니다.")

    content = response.text.strip()

    # 코드펜스나 잡텍스트가 섞여 와도 JSON 배열만 추출
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"모델 응답에서 JSON 배열을 찾을 수 없습니다:\n{content}")
    json_str = content[start : end + 1]
    data = json.loads(json_str)
    return data


def pdf_to_images(pdf_bytes, zoom=2.0):
    """PDF의 각 페이지를 PNG 이미지 바이트로 변환한다."""
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def is_pdf_file(uploaded_file):
    return uploaded_file.name.lower().endswith(".pdf") or uploaded_file.type == "application/pdf"


def set_word_bank(rows):
    """rows: list of {word, meaning} -> 번호 자동 부여 후 저장"""
    df = pd.DataFrame(rows)
    df = df.rename(columns={"word": "영어단어", "meaning": "한글뜻"})
    df.insert(0, "번호", range(1, len(df) + 1))
    st.session_state.word_bank = df


def renumber(df):
    df = df.reset_index(drop=True)
    df = df.drop(columns=["번호"], errors="ignore")
    df.insert(0, "번호", range(1, len(df) + 1))
    return df


def go(page):
    st.session_state.page = page


# ============================================================
# 사이드바
# ============================================================
def sidebar():
    with st.sidebar:
        st.header("⚙️ 설정")
        key_input = st.text_input(
            "Google Gemini API Key",
            value=st.session_state.api_key,
            type="password",
            help="이미지/PDF에서 단어를 추출하고 뜻을 생성할 때 사용됩니다. "
            "aistudio.google.com/apikey 에서 무료로 발급받을 수 있어요.",
        )
        st.session_state.api_key = key_input.strip()
        if key_input and not key_input.strip().isascii():
            st.warning(
                "API 키에 한글 등 특수 문자가 섞여 있는 것 같아요. "
                "키 입력창을 비우고 영문/숫자로만 된 키를 다시 붙여넣어 주세요."
            )

        st.markdown("---")
        st.subheader("진행 단계")
        steps = [
            ("register", "1️⃣ 단어장 등록"),
            ("setup", "2️⃣ 테스트 설정"),
            ("test", "3️⃣ 받아쓰기 진행"),
            ("result", "4️⃣ 정답 확인"),
        ]
        for key, label in steps:
            disabled = False
            if key == "setup" and st.session_state.word_bank.empty:
                disabled = True
            if key in ("test", "result") and not st.session_state.test_questions:
                disabled = True
            if st.button(label, key=f"nav_{key}", disabled=disabled, use_container_width=True):
                go(key)
                st.rerun()

        st.markdown("---")
        if st.button("🔄 처음부터 다시 시작", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ============================================================
# 1) 단어장 등록 페이지
# ============================================================
def page_register():
    st.title("✏️ 단어장 등록")
    st.caption("단어장 사진을 올리면 영어 단어와 한글 뜻을 자동으로 만들어 드려요.")

    tab1, tab2 = st.tabs(["📷 사진으로 등록 (OCR)", "📄 CSV 파일 불러오기"])

    with tab1:
        uploaded_file = st.file_uploader(
            "단어장 사진 또는 PDF 업로드",
            type=["png", "jpg", "jpeg", "pdf"],
            key="img_uploader",
        )

        pdf_pages = None
        if uploaded_file is not None:
            if is_pdf_file(uploaded_file):
                try:
                    pdf_pages = pdf_to_images(uploaded_file.getvalue())
                    st.caption(f"PDF에서 {len(pdf_pages)}페이지를 찾았어요.")
                    cols = st.columns(min(len(pdf_pages), 4) or 1)
                    for i, page_bytes in enumerate(pdf_pages):
                        with cols[i % len(cols)]:
                            st.image(page_bytes, caption=f"{i + 1}페이지", use_container_width=True)
                except Exception as e:
                    st.error(f"PDF를 여는 중 오류가 발생했습니다: {e}")
            else:
                st.image(uploaded_file, caption="업로드한 이미지", use_container_width=True)

        if st.button("🔍 단어 추출하기", type="primary"):
            if uploaded_file is None:
                st.error("먼저 이미지나 PDF를 업로드해주세요.")
            elif not get_api_key():
                st.error("사이드바에 Google Gemini API Key를 입력해주세요.")
            elif not get_api_key().isascii():
                st.error(
                    "API 키에 한글 등 특수 문자가 섞여 있어요. 사이드바에서 키를 지우고 "
                    "영문/숫자로만 된 키를 다시 붙여넣은 뒤 시도해주세요."
                )
            else:
                with st.spinner("파일에서 단어를 읽고 뜻을 만드는 중이에요..."):
                    try:
                        if is_pdf_file(uploaded_file):
                            pages = pdf_pages if pdf_pages is not None else pdf_to_images(
                                uploaded_file.getvalue()
                            )
                            rows = []
                            for page_bytes in pages:
                                rows.extend(
                                    extract_words_from_image(
                                        page_bytes, get_api_key(), mime_type="image/png"
                                    )
                                )
                        else:
                            rows = extract_words_from_image(
                                uploaded_file.getvalue(), get_api_key()
                            )
                        set_word_bank(rows)
                        st.success(f"{len(rows)}개의 단어를 추출했어요!")
                    except Exception as e:
                        st.error(f"추출 중 오류가 발생했습니다: {e}")

    with tab2:
        csv_file = st.file_uploader(
            "번호,영어단어,한글뜻 형식의 CSV 파일", type=["csv"], key="csv_uploader"
        )
        if csv_file is not None and st.button("📥 CSV 불러오기"):
            try:
                df = pd.read_csv(csv_file)
                if not {"영어단어", "한글뜻"}.issubset(df.columns):
                    st.error("CSV에 '영어단어', '한글뜻' 컬럼이 있어야 합니다.")
                else:
                    df = renumber(df[["영어단어", "한글뜻"]])
                    st.session_state.word_bank = df
                    st.success(f"{len(df)}개의 단어를 불러왔어요!")
            except Exception as e:
                st.error(f"CSV를 읽는 중 오류가 발생했습니다: {e}")

    st.markdown("---")

    if not st.session_state.word_bank.empty:
        st.subheader("📋 단어 목록 (수정 가능)")
        edited = st.data_editor(
            st.session_state.word_bank,
            num_rows="dynamic",
            use_container_width=True,
            key="word_editor",
            column_config={"번호": st.column_config.NumberColumn(disabled=True)},
        )
        if st.button("💾 수정 내용 저장"):
            cleaned = edited.dropna(subset=["영어단어", "한글뜻"])
            st.session_state.word_bank = renumber(cleaned)
            st.success("저장되었습니다.")
            st.rerun()

        st.markdown("#### 📤 내보내기")
        col1, col2 = st.columns(2)
        with col1:
            csv_data = st.session_state.word_bank.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ CSV로 다운로드", data=csv_data, file_name="단어장.csv", mime="text/csv"
            )
        with col2:
            st.caption("아래 표를 복사해서 구글 시트에 붙여넣으세요 (오른쪽 상단 복사 아이콘 클릭)")
        tsv = st.session_state.word_bank.to_csv(index=False, sep="\t")
        st.code(tsv, language=None)

        st.markdown("---")
        if st.button("➡️ 다음: 테스트 설정하기", type="primary"):
            go("setup")
            st.rerun()
    else:
        st.info("아직 등록된 단어가 없어요. 위에서 사진을 올리거나 CSV를 불러오세요.")


# ============================================================
# 2) 테스트 설정 페이지
# ============================================================
def page_setup():
    st.title("📝 받아쓰기 테스트 설정")

    df = st.session_state.word_bank
    if df.empty:
        st.warning("먼저 단어장을 등록해주세요.")
        if st.button("⬅️ 단어장 등록으로 이동"):
            go("register")
            st.rerun()
        return

    min_id, max_id = int(df["번호"].min()), int(df["번호"].max())
    st.info(f"등록된 단어 번호 범위: {min_id} ~ {max_id} (총 {len(df)}개)")

    col1, col2 = st.columns(2)
    with col1:
        start_id = st.number_input("시작 번호", min_value=min_id, max_value=max_id, value=min_id)
    with col2:
        end_id = st.number_input("끝 번호", min_value=min_id, max_value=max_id, value=max_id)

    available = df[(df["번호"] >= start_id) & (df["번호"] <= end_id)]
    max_q = max(1, len(available))
    num_questions = st.number_input(
        "문제 수", min_value=1, max_value=max_q, value=min(10, max_q)
    )

    st.caption(f"선택한 구간에는 {len(available)}개의 단어가 있어요.")

    st.markdown("---")
    st.markdown("### 🎧 문제 종류")
    question_type = st.radio(
        "어떤 방식으로 문제를 낼까요?",
        ["영어 발음만", "영어 + 한글 조합", "한글 뜻만"],
        index=1,
        horizontal=True,
        label_visibility="collapsed",
    )
    en_ratio = 100
    if question_type == "한글 뜻만":
        en_ratio = 0
    elif question_type == "영어 + 한글 조합":
        en_ratio = st.slider("영어 발음 비율 (%)", min_value=0, max_value=100, value=80, step=5)
        st.caption(f"영어 발음 {en_ratio}% / 한글 뜻 {100 - en_ratio}% 비율로 출제됩니다.")

    st.markdown("### ✏️ 답 입력창")
    answer_box_choice = st.radio(
        "화면에 정답 입력창을 보여줄까요?",
        ["노출 안 함 (노트에 직접 적기)", "노출함 (화면에 입력)"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    show_answer_box = answer_box_choice == "노출함 (화면에 입력)"

    st.markdown("### ⏱️ 제한 시간")
    time_choice = st.radio(
        "문제당 제한 시간을 사용할까요?",
        ["시간 설정", "시간 제한 없음"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    time_limit_seconds = None
    if time_choice == "시간 설정":
        time_limit_seconds = st.number_input(
            "문제당 제한 시간(초)", min_value=5, max_value=120, value=QUESTION_SECONDS, step=5
        )

    st.markdown("---")

    if st.button("🎯 테스트 시작하기", type="primary"):
        if start_id > end_id:
            st.error("시작 번호는 끝 번호보다 작거나 같아야 해요.")
        elif len(available) == 0:
            st.error("선택한 구간에 단어가 없어요.")
        else:
            sampled = available.sample(n=int(num_questions)).to_dict("records")
            questions = []
            for row in sampled:
                mode = random.choices(["en", "kr"], weights=[en_ratio, 100 - en_ratio])[0]
                questions.append(
                    {
                        "id": row["번호"],
                        "word": row["영어단어"],
                        "meaning": row["한글뜻"],
                        "mode": mode,
                    }
                )
            st.session_state.test_questions = questions
            st.session_state.current_q = 0
            st.session_state.question_start_time = None
            st.session_state.played_q = -1
            st.session_state.user_answers = {}
            st.session_state.test_config = {
                "show_answer_box": show_answer_box,
                "time_limit_seconds": time_limit_seconds,
            }
            go("test")
            st.rerun()


# ============================================================
# 3) 받아쓰기 테스트 진행 페이지
# ============================================================
def advance_question():
    st.session_state.current_q += 1
    st.session_state.question_start_time = None
    if st.session_state.current_q >= len(st.session_state.test_questions):
        go("result")


def page_test():
    questions = st.session_state.test_questions
    total = len(questions)
    idx = st.session_state.current_q
    config = st.session_state.test_config
    time_limit = config.get("time_limit_seconds")
    show_answer_box = config.get("show_answer_box", False)

    if idx >= total:
        go("result")
        st.rerun()
        return

    q = questions[idx]

    st.title("🔊 받아쓰기 테스트")
    st.progress((idx) / total, text=f"{idx + 1} / {total} 번 문제")

    remaining = None
    if time_limit:
        # 1초마다 자동 새로고침 (카운트다운용)
        st_autorefresh(interval=1000, key=f"autorefresh_{idx}")

        if st.session_state.question_start_time is None:
            st.session_state.question_start_time = time.time()

        elapsed = time.time() - st.session_state.question_start_time
        remaining = max(0, time_limit - int(elapsed))

        st.metric("⏳ 남은 시간(초)", remaining)
        st.progress(remaining / time_limit)

    st.markdown("---")

    replay_clicked = st.button("🔊 다시 듣기", key=f"replay_{idx}")

    where = "화면 입력창에" if show_answer_box else "노트에"
    if q["mode"] == "en":
        st.markdown(f"### 잘 듣고 {where} 영어 스펠링을 적어보세요!")
        audio_bytes = get_audio_bytes(q["word"], lang="en")
    else:
        st.markdown(f"### 뜻을 보고(듣고) {where} 영어 단어를 적어보세요!")
        st.markdown(f"<div class='big-word'>{q['meaning']}</div>", unsafe_allow_html=True)
        audio_bytes = get_audio_bytes(q["meaning"], lang="ko")

    # 1초마다 오는 자동 새로고침 중에도 오디오 엘리먼트가 사라져 재생이
    # 끊기지 않도록, 매 rerun마다 동일한 위치에서 항상 렌더링한다.
    # autoplay는 이 문제가 처음 나타났을 때(또는 다시 듣기 클릭 시)만 True로 준다.
    should_autoplay = replay_clicked or st.session_state.played_q != idx
    st.audio(audio_bytes, format="audio/mp3", autoplay=should_autoplay)
    st.caption("소리가 자동으로 나오지 않으면 위 재생 버튼이나 '🔊 다시 듣기'를 눌러주세요.")
    if should_autoplay:
        st.session_state.played_q = idx

    st.markdown("---")

    if show_answer_box:
        # 정답 입력창 노출: Enter 또는 버튼으로 제출하면서 입력한 답을 저장한다.
        with st.form(key=f"next_form_{idx}", clear_on_submit=True):
            answer = st.text_input(
                "정답을 입력하세요 (영어 스펠링)",
                key=f"answer_input_{idx}",
                placeholder="여기에 정답을 입력하고 Enter ↵ (또는 아래 버튼 클릭)",
            )
            submitted = st.form_submit_button("➡️ 다음 문제", type="primary", use_container_width=True)
        if submitted:
            st.session_state.user_answers[idx] = answer
            advance_question()
            st.rerun()
    else:
        # 입력창 미노출: 노트에 직접 적고 버튼으로만 다음 문제로 이동한다.
        if st.button("➡️ 다음 문제", type="primary", use_container_width=True, key=f"next_btn_{idx}"):
            advance_question()
            st.rerun()

    if remaining is not None and remaining <= 0:
        advance_question()
        st.rerun()


# ============================================================
# 4) 최종 정답 확인 페이지
# ============================================================
def page_result():
    st.title("✅ 정답 확인")

    questions = st.session_state.test_questions
    show_answer_box = st.session_state.test_config.get("show_answer_box", False)
    user_answers = st.session_state.user_answers

    if show_answer_box:
        st.caption("화면에 입력한 답을 자동으로 채점했어요.")
    else:
        st.caption("노트에 적은 답과 아래 정답을 비교하며 채점해보세요.")

    rows = []
    correct_count = 0
    for i, q in enumerate(questions):
        mode_label = "🔊 영어 음성" if q["mode"] == "en" else "🇰🇷 한글 뜻"
        row = {
            "문제 번호": i + 1,
            "출제 방식": mode_label,
            "정답(영어 스펠링)": q["word"],
            "정답(한글 뜻)": q["meaning"],
        }
        if show_answer_box:
            user_answer = (user_answers.get(i) or "").strip()
            is_correct = user_answer.lower() == q["word"].strip().lower()
            if is_correct:
                correct_count += 1
            row["내가 입력한 답"] = user_answer
            row["채점"] = "⭕ 정답" if is_correct else "❌ 오답"
        rows.append(row)
    result_df = pd.DataFrame(rows)

    if show_answer_box:
        st.metric("점수", f"{correct_count} / {len(questions)}")

    st.dataframe(result_df, use_container_width=True, hide_index=True)

    score_hint = st.columns(2)
    with score_hint[0]:
        csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 정답표 CSV 다운로드", data=csv_data, file_name="정답표.csv", mime="text/csv")
    with score_hint[1]:
        if st.button("🔁 새로운 테스트 시작하기", type="primary", use_container_width=True):
            st.session_state.test_questions = []
            st.session_state.current_q = 0
            st.session_state.question_start_time = None
            st.session_state.user_answers = {}
            go("setup")
            st.rerun()


# ============================================================
# 라우팅
# ============================================================
# 사이드바는 현재 페이지의 상태 변경이 모두 반영된 뒤 그려야
# "다음 단계로 이동" 직후에도 버튼 활성화 상태가 즉시 맞다.
if st.session_state.page == "register":
    page_register()
elif st.session_state.page == "setup":
    page_setup()
elif st.session_state.page == "test":
    page_test()
elif st.session_state.page == "result":
    page_result()

sidebar()
