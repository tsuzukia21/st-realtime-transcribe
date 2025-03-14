import logging
import queue
import time
import os
import tempfile
import asyncio
import pydub
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import requests
import altair as alt
import pandas as pd

logger = logging.getLogger(__name__)

# セッション状態の初期化
if 'recording' not in st.session_state:
    st.session_state.recording = False  # 録音中かどうかのフラグ
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None  # 録音済み音声データ
if 'temp_audio_file' not in st.session_state:
    st.session_state.temp_audio_file = None  # 一時保存用の音声ファイルパス
if 'is_capturing' not in st.session_state:
    st.session_state.is_capturing = False  # 音声キャプチャ中かどうかのフラグ
if 'capture_buffer' not in st.session_state:
    st.session_state.capture_buffer = pydub.AudioSegment.empty()  # 音声キャプチャ用バッファ
if 'volume_history' not in st.session_state:
    st.session_state.volume_history = []  # 音量履歴（グラフ表示用）

# 録音開始/停止ボタン
def toggle_recording():
    if st.session_state.recording:
        st.toast(f"**録音停止**",icon=":material/mic_off:")
    else:
        st.toast(f"**録音開始**",icon=":material/mic:")
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
    type="primary" if st.session_state.recording else "secondary"
)

if 'full_text' not in st.session_state:
    st.session_state.full_text = ""

# 無音部分の検出結果を表示する場所
silence_info_placeholder = st.empty()

# 無音検出用のパラメータ設定
# サイドバーにスライダーを配置して、ユーザーがリアルタイムに調整できるようにする
st.sidebar.title("無音検出設定")
silence_threshold = st.sidebar.slider(
    "無音しきい値 (dB)", 
    -80, 0, -35,
    help="音声を「無音」と判断する音量レベルを設定します。\n"
         "値が小さいほど（例：-50dB）より小さな音も「音声あり」と判断します。\n"
         "値が大きいほど（例：-20dB）大きな音のみを「音声あり」と判断します。"
)

min_silence_duration = st.sidebar.slider(
    "最小無音時間 (ms)", 
    100, 500, 200,
    help="この時間以上の無音が続いた場合に「無音区間」と判断します。\n"
         "短すぎると話の途中の短い間も無音と判断され、\n"
         "長すぎると長めの間も音声の一部と判断されます。"
)

# 録音設定
st.sidebar.title("録音設定")
auto_stop_duration = st.sidebar.slider(
    "無音検出時の自動停止 (ms)", 
    100, 2000, 1000,
    help="この時間以上の無音が続くと、自動的に録音を停止します。\n"
         "話者の発話が終わったことを検出するための設定です。\n"
         "短すぎると話の途中で録音が止まり、長すぎると無駄な無音時間が録音されます。"
)

min_recording_duration = st.sidebar.slider(
    "最低録音時間 (秒)", 
    1, 10, 2,
    help="録音を保存する最低限の長さを設定します。\n"
         "これより短い録音は無視されます。\n"
         "短すぎると雑音なども録音されやすく、長すぎると短い返事なども無視されます。"
)
with st.sidebar:
    language = st.selectbox(
        "言語",
        ["ja", "en" ,"zh"],
        index=0,
        help="音声認識に使用する言語を選択します。"
    )
    if st.button("initial promptのリセット",type="primary"):
        st.session_state.full_text = ""

status_placeholder = st.empty()
chart_placeholder = st.empty()
rec_status_placeholder = st.empty()
transcription_placeholder = st.empty()

# 過去の音声データを保持するバッファ
sound_window_len = 5000  # 5秒
sound_window_buffer = None

# 一時ファイルへの保存とオーディオプレーヤーの表示
# 非同期関数として実装し、UIのブロッキングを防止
async def save_and_display_audio(audio_segment):
    # 録音時間のチェック - 短すぎる録音は処理しない
    recording_duration = len(audio_segment) / 1000.0  # ミリ秒から秒に変換
    
    if recording_duration < st.session_state.get('min_recording_duration', min_recording_duration):
        # 最低録音時間未満の場合は処理を中断
        rec_status_placeholder.empty()
        return
    rec_status_placeholder.success("サーバーへ送信!!",icon=":material/check_circle:")
    
    # 一時ファイルの作成 - システムの一時ディレクトリに保存
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_file_path = temp_file.name
    temp_file.close()
    
    # 音声ファイルの保存 - 非同期処理で実行してUIブロッキングを防止
    await asyncio.to_thread(audio_segment.export, temp_file_path, format="wav")
    
    # 以前の一時ファイルがあれば削除 - リソースリークを防止
    if st.session_state.temp_audio_file and os.path.exists(st.session_state.temp_audio_file):
        try:
            await asyncio.to_thread(os.unlink, st.session_state.temp_audio_file)
        except Exception as e:
            logger.warning(f"一時ファイルの削除に失敗: {e}")
    
    # 新しい一時ファイルのパスを保存
    st.session_state.temp_audio_file = temp_file_path
    
    # メモリ効率化: オーディオセグメントをコピーする代わりに参照を保存
    st.session_state.recorded_audio = audio_segment
    
    # 録音後に不要なバッファをクリア - メモリ使用量を削減
    global sound_window_buffer
    sound_window_buffer = None
    
    # GCを強制的に実行してメモリを解放 - 大きな音声データを扱うため必要なケースがある
    import gc
    gc.collect()
    
    try:
        # 音声認識APIへのリクエスト送信
        # ファイルを開いてバイナリモードで読み込む
        with open(temp_file_path, 'rb') as audio_file:
            response = requests.post(
                "http://XXX.XXX.XXX.XXX:XXXX/transcribe",  # 音声認識APIのエンドポイント
                files={"audio": ('audio.wav', audio_file, 'audio/wav')},
                data={"model": "汎用モデル", "save_audio": False, "file_name": "temp_audio.wav","language" : language,
                      "initial_prompt": st.session_state.full_text}  # 初期プロンプトに前の認識結果を使用
            )
        
        # レスポンスのステータスコードを確認
        if response.status_code == 200:
            try:
                json_data = response.json()
                if "full_text" in json_data:
                    full_text = json_data['full_text']
                    # 文字列の先頭に追加（新しいテキストが上に表示される）
                    st.session_state.full_text = full_text + "\n" + st.session_state.full_text
                else:
                    st.markdown(f"**APIレスポンス:**\n{json_data}")
            except ValueError:
                st.error(f"レスポンスをJSONとして解析できません: {response.text[:100]}...")
        else:
            st.error(f"APIエラー: ステータスコード {response.status_code}")
            st.code(response.text[:200]) # エラーメッセージの最初の部分を表示
    except Exception as e:
        st.error(f"APIリクエストエラー: {str(e)}")
        logger.error(f"API通信中にエラーが発生: {e}", exc_info=True)

    # APIを使用した場合
    # with open(temp_file_path, 'rb') as audio_file:
    #     client = OpenAI()
    #     response = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    #     full_text = response.text
    #     st.session_state.full_text = full_text + "\n" + st.session_state.full_text

    transcription_placeholder.markdown(st.session_state.full_text)
    
    # セッション状態に最低録音時間を保存
    st.session_state.min_recording_duration = min_recording_duration

# 最後に音が検出された時間を記録
last_sound_time = time.time()
no_sound_duration = 0

# 呼び出し側の関数も非同期にする
async def process_audio():
    await save_and_display_audio(st.session_state.capture_buffer)
    
# 非同期関数を実行するためのヘルパー関数
# Streamlitの実行環境では非同期処理の扱いが複雑なため、この関数で適切に管理する
def run_async(async_func):
    import asyncio
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

# オーディオ処理のメインループを非同期関数化
async def process_audio_stream(webrtc_ctx):
    sound_window_buffer = None
    
    while True:  # メインループ - WebRTCストリームから音声を継続的に処理
        if webrtc_ctx.audio_receiver:
            try:
                # タイムアウト付きでフレームを取得 - ブロッキングを防止
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            # 受信した音声フレームを処理しchunkを作成
            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                # 音声バッファの管理 - 指定サイズの履歴を保持
                if sound_window_buffer is None:
                    sound_window_buffer = sound_chunk
                else:
                    sound_window_buffer += sound_chunk
                
                # バッファが指定の長さを超えたら、古い部分を削除（スライディングウィンドウ）
                if len(sound_window_buffer) > sound_window_len:
                    sound_window_buffer = sound_window_buffer[-sound_window_len:]
                
                # 現在の音量レベル計算
                current_db = sound_chunk.dBFS if len(sound_chunk) > 0 else -100
                
                # 音量履歴の更新とグラフ表示
                st.session_state.volume_history.append({"音量": current_db})
                if len(st.session_state.volume_history) > 100:
                    st.session_state.volume_history.pop(0)  # 100ポイントに制限
                
                # データがリスト形式の場合、DataFrameに変換
                df = pd.DataFrame(st.session_state.volume_history)
                df = df.reset_index().rename(columns={"index": "時間"})

                # x軸 (時間) を非表示にする設定
                chart = alt.Chart(df).mark_line().encode(
                    x=alt.X("時間", axis=None),  # x軸を非表示にする
                    y=alt.Y("音量", title="音量 (dB)")
                ).properties(
                    height=200,
                    width='container'
                )

                # st.altair_chart の代わりにプレースホルダーを使って更新
                chart_placeholder.altair_chart(chart, use_container_width=True)
                
                # 無音部分の検出と状態管理
                if sound_window_buffer:
                    if len(sound_chunk) > 0:
                        silence_info = f"\n現在の音量: {current_db:.2f} dB"
                        
                        # 音量に応じて状態を表示と録音制御の判断
                        if current_db <= silence_threshold:  # 無音状態
                            status_placeholder.info("無音状態です",icon=":material/sentiment_calm:")
                            if st.session_state.is_capturing:
                                no_sound_duration += len(sound_chunk)  # 無音継続時間を計測
                            # 録音中だが音声キャプチャしていない場合のメッセージを表示
                            elif st.session_state.recording and not st.session_state.is_capturing:
                                rec_status_placeholder.info("音声の入力を待っています",icon=":material/sentiment_calm:")
                        else:  # 音声検出状態
                            status_placeholder.success("音声を検出しています",icon=":material/check_circle:")
                            no_sound_duration = 0  # 無音継続時間をリセット
                            
                            # 録音開始ロジック - 音声検出時に自動的にキャプチャ開始
                            if st.session_state.recording and not st.session_state.is_capturing:
                                st.session_state.is_capturing = True
                                st.session_state.capture_buffer = pydub.AudioSegment.empty()
                                rec_status_placeholder.warning("音声をキャプチャ中...",icon=":material/mic:")
                    
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
                            
        else:
            # WebRTC接続待機状態
            status_placeholder.warning("音声の受信を待っています...",icon=":material/pending:")
            time.sleep(0.1)

# メインループを非同期で開始する
run_async(process_audio_stream(webrtc_ctx))

# アプリケーション終了時の一時ファイル削除
if st.session_state.temp_audio_file and os.path.exists(st.session_state.temp_audio_file):
    try:
        os.unlink(st.session_state.temp_audio_file)
    except Exception as e:
        logger.warning(f"一時ファイルの削除に失敗: {e}")
