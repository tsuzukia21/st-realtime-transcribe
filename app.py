import asyncio
import logging
import os
import queue
import tempfile
import time
import altair as alt
import pandas as pd
import pydub
import requests
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(layout="wide", page_title="realtime-transcribe")

logger = logging.getLogger(__name__)

# 設定値（定数）
SERVER_ENDPOINT = "http://localhost:5000/transcribe"  # 音声認識APIのエンドポイント
SOUND_WINDOW_LEN = 5000  # 過去音声バッファの保持長(ms) = 5秒
VOLUME_HISTORY_LIMIT = 100  # 音量履歴グラフの最大ポイント数

# 文字起こしバックエンドの選択
# False: 自前のサーバー(server.py)を使用 / True: OpenAI Whisper API を使用
USE_API = False

# セッション状態の初期化
DEFAULT_SESSION_STATE = {
    "recording": False,                            # 録音中かどうかのフラグ
    "recorded_audio": None,                        # 録音済み音声データ
    "is_capturing": False,                         # 音声キャプチャ中かどうかのフラグ
    "capture_buffer": pydub.AudioSegment.empty(),  # 音声キャプチャ用バッファ
    "volume_history": [],                          # 音量履歴（グラフ表示用）
    "full_text": (),                               # 文字起こし結果（新しいものが先頭）
}
for key, default in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default

# 録音開始/停止ボタン
def toggle_recording():
    if st.session_state.recording:
        st.toast("**録音停止**", icon=":material/mic_off:")
    else:
        st.toast("**録音開始**", icon=":material/mic:")
    st.session_state.recording = not st.session_state.recording
    if not st.session_state.recording:
        # 録音停止時、キャプチャも停止
        st.session_state.is_capturing = False

webrtc_ctx = webrtc_streamer(
    key="sendonly-audio",  # WebRTCコンポーネントの一意の識別子
    mode=WebRtcMode.SENDONLY,  # 送信専用モード
    audio_receiver_size=256,  # 受信バッファサイズ
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],  # STUN/TURNサーバー設定
        "iceTransportPolicy": "all",  # ICE接続ポリシー
    },
    media_stream_constraints={"audio": True},  # 音声のみ有効化
)

st.button(
    "録音 " + ("停止" if st.session_state.recording else "開始"),
    on_click=toggle_recording,
    type="primary" if st.session_state.recording else "secondary",
    disabled=webrtc_ctx.audio_receiver is None,
)

if webrtc_ctx.audio_receiver is None:
    st.session_state.recording = False

# 無音部分の検出結果を表示する場所
silence_info_placeholder = st.empty()

# 無音検出用のパラメータ設定
# サイドバーにスライダーを配置して、ユーザーがリアルタイムに調整できるようにする
st.sidebar.title("無音検出設定")
silence_threshold = st.sidebar.slider(
    "無音しきい値 (dB)",
    -80, 0, -35,
    disabled=st.session_state.recording,
    help="音声を「無音」と判断する音量レベルを設定します。\n"
         "値が小さいほど（例：-50dB）より小さな音も「音声あり」と判断します。\n"
         "値が大きいほど（例：-20dB）大きな音のみを「音声あり」と判断します。"
)

min_silence_duration = st.sidebar.slider(
    "最小無音時間 (ms)",
    100, 500, 200,
    disabled=st.session_state.recording,
    help="この時間以上の無音が続いた場合に「無音区間」と判断します。\n"
         "短すぎると話の途中の短い間も無音と判断され、\n"
         "長すぎると長めの間も音声の一部と判断されます。"
)

# 録音設定
st.sidebar.title("録音設定")
auto_stop_duration = st.sidebar.slider(
    "無音検出時の自動停止 (ms)",
    100, 2000, 1000,
    disabled=st.session_state.recording,
    help="この時間以上の無音が続くと、自動的に録音を停止します。\n"
         "話者の発話が終わったことを検出するための設定です。\n"
         "短すぎると話の途中で録音が止まり、長すぎると無駄な無音時間が録音されます。"
)

min_recording_duration = st.sidebar.slider(
    "最低録音時間 (秒)",
    1, 10, 2,
    disabled=st.session_state.recording,
    help="録音を保存する最低限の長さを設定します。\n"
         "これより短い録音は無視されます。\n"
         "短すぎると雑音なども録音されやすく、長すぎると短い返事なども無視されます。"
)
with st.sidebar:
    language = st.selectbox(
        "言語",
        ["ja", "en", "zh"],
        index=0,
        disabled=st.session_state.recording,
        help="音声認識に使用する言語を選択します。"
    )
    if st.button("initial prompt/文字起こし履歴のリセット", type="primary", disabled=st.session_state.recording):
        st.session_state.full_text = ()

status_placeholder = st.empty()
chart_placeholder = st.empty()
rec_status_placeholder = st.empty()
transcription_placeholder = st.empty()

def render_transcription():
    """文字起こし結果（新しいものが先頭）をプレースホルダに表示する。"""
    text = "\n".join(st.session_state.full_text) if st.session_state.full_text else ""
    transcription_placeholder.markdown(text)


def transcribe_via_server(temp_file_path):
    """自前のサーバー(server.py)で文字起こしする。結果テキスト、失敗時は None を返す。"""
    with open(temp_file_path, "rb") as audio_file:
        response = requests.post(
            SERVER_ENDPOINT,
            files={"audio": ("audio.wav", audio_file, "audio/wav")},
            data={
                "language": language,
                "initial_prompt": "\n".join(st.session_state.full_text) if st.session_state.full_text else "",
            },
        )

    if response.status_code != 200:
        st.error(f"APIエラー: ステータスコード {response.status_code}")
        st.code(response.text[:200])  # エラーメッセージの最初の部分を表示
        return None

    try:
        json_data = response.json()
    except ValueError:
        st.error(f"レスポンスをJSONとして解析できません: {response.text[:100]}...")
        return None

    if "full_text" in json_data:
        return json_data["full_text"]

    st.markdown(f"**APIレスポンス:**\n{json_data}")
    return None


def transcribe_via_api(temp_file_path):
    """OpenAI Whisper API で文字起こしする。結果テキストを返す。"""
    from openai import OpenAI

    with open(temp_file_path, "rb") as audio_file:
        client = OpenAI()
        response = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return response.text

# 一時ファイルへの保存と音声認識リクエスト
# 非同期関数として実装し、UIのブロッキングを防止
async def save_and_display_audio(audio_segment):
    # 録音時間のチェック - 短すぎる録音は処理しない
    recording_duration = len(audio_segment) / 1000.0  # ミリ秒から秒に変換
    if recording_duration < min_recording_duration:
        # 最低録音時間未満の場合は処理を中断
        rec_status_placeholder.empty()
        return

    rec_status_placeholder.success("サーバーへ送信!!", icon=":material/check_circle:")

    # 一時ファイルの作成 - 処理後に必ず削除する
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        # 音声ファイルの保存 - 別スレッドで実行してUIブロッキングを防止
        await asyncio.to_thread(audio_segment.export, temp_file_path, format="wav")

        # メモリ効率化: オーディオセグメントは参照のみ保持
        st.session_state.recorded_audio = audio_segment

        # USE_API に応じて文字起こしバックエンドを切り替える
        # st.session_state / st.* に触れるため、別スレッドではなくこのまま実行する
        if USE_API:
            full_text = transcribe_via_api(temp_file_path)
        else:
            full_text = transcribe_via_server(temp_file_path)

        if full_text:
            # 文字列の先頭に追加（新しいテキストが上に表示される）
            st.session_state.full_text = (full_text,) + st.session_state.full_text

    except Exception as e:
        st.error(f"APIリクエストエラー: {str(e)}")
        logger.error(f"API通信中にエラーが発生: {e}", exc_info=True)

    finally:
        # 一時ファイルを削除 - リソースリークを防止
        if os.path.exists(temp_file_path):
            try:
                await asyncio.to_thread(os.unlink, temp_file_path)
            except Exception as e:
                logger.warning(f"一時ファイルの削除に失敗: {e}")

    render_transcription()


# 呼び出し側の関数も非同期にする
async def process_audio():
    await save_and_display_audio(st.session_state.capture_buffer)


# 非同期関数を実行するためのヘルパー関数
# Streamlitの実行環境では非同期処理の扱いが複雑なため、この関数で適切に管理する
def run_async(async_func):
    try:
        # 既存のループを取得 - すでに実行中のループがあれば活用
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # すでに実行中の場合は、future/taskとして追加
            # 別スレッドで実行中のループに新しいタスクを安全に追加
            future = asyncio.run_coroutine_threadsafe(async_func, loop)
            return future.result()
        else:
            # ループが存在するが実行中でない場合
            return loop.run_until_complete(async_func)
    except RuntimeError:
        # ループが存在しない場合は新規作成 - 初回実行時など
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(async_func)
        finally:
            # ループをクローズする前に保留中のタスクを完了させる
            # リソースリークを防止するための重要な後処理
            pending = asyncio.all_tasks(new_loop)
            for task in pending:
                task.cancel()
            new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            new_loop.close()


def frames_to_segment(audio_frames):
    """WebRTCの音声フレーム列を1つのAudioSegmentに変換する。"""
    sound_chunk = pydub.AudioSegment.empty()
    for audio_frame in audio_frames:
        sound = pydub.AudioSegment(
            data=audio_frame.to_ndarray().tobytes(),
            sample_width=audio_frame.format.bytes,
            frame_rate=audio_frame.sample_rate,
            channels=len(audio_frame.layout.channels),
        )
        sound_chunk += sound
    return sound_chunk


def update_volume_chart(current_db):
    """音量履歴を更新し、折れ線グラフを描画する。"""
    st.session_state.volume_history.append({"音量": current_db})
    if len(st.session_state.volume_history) > VOLUME_HISTORY_LIMIT:
        st.session_state.volume_history.pop(0)  # 上限ポイント数に制限

    df = pd.DataFrame(st.session_state.volume_history)
    df = df.reset_index().rename(columns={"index": "時間"})

    # x軸 (時間) を非表示にする設定
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("時間", axis=None),  # x軸を非表示にする
            y=alt.Y("音量", title="音量 (dB)"),
        )
        .properties(height=200, width="container")
    )
    chart_placeholder.altair_chart(chart, use_container_width=True)


# オーディオ処理のメインループを非同期関数化
async def process_audio_stream(webrtc_ctx):
    sound_window_buffer = None  # 過去音声のスライディングウィンドウ
    no_sound_duration = 0       # 無音継続時間(ms)

    while True:  # メインループ - WebRTCストリームから音声を継続的に処理
        if not webrtc_ctx.audio_receiver:
            # WebRTC接続待機状態
            status_placeholder.warning("音声の受信を待っています...", icon=":material/pending:")
            time.sleep(0.1)
            if st.session_state.full_text:
                render_transcription()
            continue

        try:
            # タイムアウト付きでフレームを取得 - ブロッキングを防止
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            logger.warning("Queue is empty. Abort.")
            break

        # 受信した音声フレームを1つのchunkに変換
        sound_chunk = frames_to_segment(audio_frames)
        if len(sound_chunk) == 0:
            if st.session_state.full_text:
                render_transcription()
            continue

        # 音声バッファの管理 - 指定サイズの履歴を保持（スライディングウィンドウ）
        if sound_window_buffer is None:
            sound_window_buffer = sound_chunk
        else:
            sound_window_buffer += sound_chunk
        if len(sound_window_buffer) > SOUND_WINDOW_LEN:
            sound_window_buffer = sound_window_buffer[-SOUND_WINDOW_LEN:]

        # 現在の音量レベル計算とグラフ更新
        current_db = sound_chunk.dBFS
        update_volume_chart(current_db)

        # 無音部分の検出と録音制御の判断
        silence_info = f"\n現在の音量: {current_db:.2f} dB"
        if current_db <= silence_threshold:  # 無音状態
            status_placeholder.info("無音状態です", icon=":material/sentiment_calm:")
            if st.session_state.is_capturing:
                no_sound_duration += len(sound_chunk)  # 無音継続時間を計測
            elif st.session_state.recording:
                # 録音中だが音声キャプチャしていない場合のメッセージを表示
                rec_status_placeholder.info("音声の入力を待っています", icon=":material/sentiment_calm:")
        else:  # 音声検出状態
            status_placeholder.success("音声を検出しています", icon=":material/check_circle:")
            no_sound_duration = 0  # 無音継続時間をリセット
            # 録音開始ロジック - 音声検出時に自動的にキャプチャ開始
            if st.session_state.recording and not st.session_state.is_capturing:
                st.session_state.is_capturing = True
                st.session_state.capture_buffer = pydub.AudioSegment.empty()
                rec_status_placeholder.warning("音声をキャプチャ中...", icon=":material/mic:")

        # 情報を表示
        silence_info_placeholder.text(silence_info)

        # 録音バッファ更新と自動停止ロジック
        if st.session_state.recording and st.session_state.is_capturing:
            st.session_state.capture_buffer += sound_chunk

            # 無音状態が一定時間続いた場合の自動停止処理
            if no_sound_duration >= auto_stop_duration:
                st.session_state.is_capturing = False
                if len(st.session_state.capture_buffer) > 0:
                    await process_audio()  # 録音データの処理と音声認識
                no_sound_duration = 0

        if st.session_state.full_text:
            render_transcription()


# メインループを非同期で開始する
run_async(process_audio_stream(webrtc_ctx))
