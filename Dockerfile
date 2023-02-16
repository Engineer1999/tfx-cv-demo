FROM python:3.7

COPY ./ ./

RUN apt update && apt install make

RUN python -m pip install --upgrade pip

RUN pip install -r Requirement.txt

CMD ["make", "run_pipeline"]

