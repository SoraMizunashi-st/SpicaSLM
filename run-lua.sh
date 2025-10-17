#!/bin/bash
# run-lua.sh

# Lua実行ファイルのディレクトリを定義
LUA_BIN_DIR="./bin/Lua/"

# 実行ファイル名を取得 (例: ./script/test.lua)
SCRIPT_FILE="$1"

# $LUA_BIN_DIR/lua を使ってスクリプトを実行
# スクリプトへの引数 ($2以降) もすべて渡す
"$LUA_BIN_DIR/lua" "$SCRIPT_FILE" "${@:2}"