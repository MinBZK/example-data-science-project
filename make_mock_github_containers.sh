# you need to login first to github with your PAT
docker pull alpine:latest
docker pull alpine:edge
docker pull alpine:3.17
docker pull alpine:3.18


docker tag alpine:latest ghcr.io/minbzk/test-aivt:latest
docker push ghcr.io/minbzk/test-aivt:latest

docker tag alpine:edge ghcr.io/minbzk/test-aivt:pr11
docker push ghcr.io/minbzk/test-aivt:pr11

docker tag alpine:3.17 ghcr.io/minbzk/test-aivt:pr12
docker push ghcr.io/minbzk/test-aivt:pr12

docker tag alpine:3.18 ghcr.io/minbzk/test-aivt:v.3.0
docker push ghcr.io/minbzk/test-aivt:v.3.0
