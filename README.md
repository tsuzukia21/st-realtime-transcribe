# Demo

![Demo](./demo.gif)

# Description

This app enables real-time speech-to-text transcription.  
You can either set up your own transcribe server or use OpenAI's API to call Whisper for transcription.  
Users can configure settings such as silent mode and recording mode to match their recording conditions.

このアプリはリアルタイム文字起こしをすることが出来ます。  
文字起こしは自身でサーバーを建てるか、OpenAIのAPIを使ってWhisperを呼び出すこともできます。  
無音設定、録音設定等、ユーザーが録音状況に合わせて設定することが出来ます。

# How to use

For details on usage and code, please refer to the following links.(Japanese)

使い方、コードの詳細は以下のリンクから確認できます。

[リアルタイム文字起こしアプリをstreamlitで作成してみよう！](https://zenn.dev/tsuzukia/articles/ca4704708c1066)

# Installation

```python
pip install -r requirements.txt
```

# Usage

```python
streamlit run app.py
python server.py
```

# API Key

Please set up the API Key as needed.

必要に応じAPI Keyの設定をしてください

# Author

* tsuzukia21
* Twitter : [https://twitter.com/tsuzukia_prgm](https://twitter.com/tsuzukia_prgm)
* zenn : https://zenn.dev/tsuzukia

Thank you!
