# This is the default base image for use with user models and workflows.
FROM datarobot/dropin-env-base-jdk:debian11-py3.9-jdk11.0.16-drum1.11.5-mlops9.2.8

# Install the list of core requirements, e.g. sklearn, numpy, pandas, flask.
# **Don't modify this file!**
COPY requirements.txt requirements.txt

# '--upgrade-strategy eager' will upgrade installed dependencies
# according to package requirements or to the latest
RUN pip3 install -U pip && \
    pip3 install -U --upgrade-strategy eager --no-cache-dir -r requirements.txt  && \
    rm -rf requirements.txt

# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here
ENV HOME=/opt CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
WORKDIR ${CODE_DIR}

ARG UID=1001
ARG GID=1001
RUN chown ${UID}:${GID} ${CODE_DIR}
RUN chmod -R 777 ${CODE_DIR}

EXPOSE 9001

COPY ./download_model.py ${CODE_DIR}/
COPY ./*.sh ${CODE_DIR}/
COPY ./nginx.conf /etc/nginx/nginx.conf

ENV WITH_ERROR_SERVER=1
ENV CLASS_LABELS_FILE=${CODE_DIR}/classLabels.txt

ENTRYPOINT ["/opt/code/start_server.sh"]

