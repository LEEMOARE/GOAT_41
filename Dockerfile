# Python 3.8 이미지를 기본 이미지로 사용
FROM python:3.8

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지를 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# 애플리케이션 코드 추가
COPY . .


# 컨테이너가 시작될 때 실행되는 명령어 설정
CMD ["python", "nih.py"]