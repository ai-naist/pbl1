# pbl1
```
# Homebrewの代替インストール
mkdir -p $HOME/.homebrew
git clone https://github.com/Homebrew/brew $HOME/.homebrew/Homebrew
mkdir -p $HOME/.homebrew/bin
ln -s $HOME/.homebrew/Homebrew/bin/brew $HOME/.homebrew/bin

# 既存のHomebrewディレクトリを確認
ls -la $HOME/.homebrew/Homebrew

# シンボリックリンクの確認と削除
rm $HOME/.homebrew/bin/brew
ln -s $HOME/.homebrew/Homebrew/bin/brew $HOME/.homebrew/bin

# 環境変数の設定
export PATH="$HOME/.homebrew/bin:$PATH"
export HOMEBREW_PREFIX="$HOME/.homebrew"
export HOMEBREW_CELLAR="$HOME/.homebrew/Cellar"
export HOMEBREW_REPOSITORY="$HOME/.homebrew/Homebrew"
export HOMEBREW_NO_ANALYTICS=1

source ~/.zshrc
brew --version

# llvmのインストール
brew install llvm

# ryeの環境にclangを認識させる
export PATH="$HOME/.homebrew/opt/llvm/bin:$PATH"
export LDFLAGS="-L$HOME/.homebrew/opt/llvm/lib"
export CPPFLAGS="-I$HOME/.homebrew/opt/llvm/include"
export CC=clang
export CXX=clang++

rye sync


# モニタリング
vmstat
ps aux | grep python

```