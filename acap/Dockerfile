ARG ARCH=aarch64
ARG VERSION=12.11.0
ARG UBUNTU_VERSION=24.04

FROM axisecp/acap-native-sdk:${VERSION}-${ARCH}-ubuntu${UBUNTU_VERSION}

COPY ./app /opt/app/
WORKDIR /opt/app

# Fix WSL / Windows-mount file permissions (777 silently breaks web config on device)
RUN find . -type f -exec chmod 644 {} + && \
    find . -type d -exec chmod 755 {} +

RUN . /opt/axis/acapsdk/environment-setup* && \
    acap-build ./ -a model/traindet.tflite -a model/yamnet_class_map.csv
