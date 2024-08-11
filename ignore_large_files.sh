#!/bin/bash

# 설정할 파일 크기 (여기서는 50MB)
MAX_SIZE=50000000

# 현재 디렉토리의 파일 중 설정된 크기 이상의 파일을 .gitignore에 추가
find . -type f -size +${MAX_SIZE}c -exec echo "{}" >> .gitignore \;

# 중복 제거
sort -u -o .gitignore .gitignore
