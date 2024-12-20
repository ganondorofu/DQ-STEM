import http.server
import socketserver
import os

# サーバーのポート番号
PORT = 8000
# HTMLファイルのあるディレクトリ
directory = os.path.dirname(os.path.abspath("furusato_nouzei_map.html"))

# ディレクトリを移動
os.chdir(directory)

# サーバーを立てる
handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", PORT), handler) as httpd:
    print(f"サーバーが起動しました。LAN内から次のURLでアクセスできます:")
    print(f"http://{os.getenv('COMPUTERNAME', 'localhost')}:{PORT}/furusato_nouzei_map.html")
    print("終了するには Ctrl+C を押してください。")
    httpd.serve_forever()
