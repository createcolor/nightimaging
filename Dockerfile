FROM python:3.7

COPY requirements.txt .
RUN python -m pip install --no-cache -r requirements.txt

COPY . /nightimaging
WORKDIR /nightimaging

ENTRYPOINT python -m demo.process_pngs -p /data -ie gw -tm Flash