#!/bin/zsh
cd /Users/dmitrii_andriianov/Documents/CUB/AI-Research-Digest-Agent
exec /opt/homebrew/Caskroom/miniconda/base/envs/ai-research-digest/bin/python /Users/dmitrii_andriianov/Documents/CUB/AI-Research-Digest-Agent/label_papers.py --sample /Users/dmitrii_andriianov/Documents/CUB/AI-Research-Digest-Agent/eval/runs/weekly-20261118/blind-sample.json --db /Users/dmitrii_andriianov/Documents/CUB/AI-Research-Digest-Agent/eval/runs/weekly-20261118/labels.sqlite3 --page /Users/dmitrii_andriianov/Documents/CUB/AI-Research-Digest-Agent/labeling/static/weekly.html --port 8766
