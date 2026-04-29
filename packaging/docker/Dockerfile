FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG LILBEE_VERSION
RUN test -n "${LILBEE_VERSION}" || (echo "LILBEE_VERSION build arg required" && exit 1) && \
    curl -fsSL -o /usr/local/bin/lilbee \
        "https://github.com/tobocop2/lilbee/releases/download/v${LILBEE_VERSION}/lilbee-linux-x86_64" && \
    chmod +x /usr/local/bin/lilbee && \
    /usr/local/bin/lilbee --version

RUN useradd -m -u 1000 lilbee
USER lilbee
WORKDIR /home/lilbee/data
ENV LILBEE_DATA_DIR=/home/lilbee/data
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/lilbee"]
CMD ["--help"]
