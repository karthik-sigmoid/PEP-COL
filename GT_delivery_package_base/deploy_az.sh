 docker buildx build --platform linux/amd64 --no-cache -t dash-prod-gt-eg:7.0 -f Dockerfile.prod .
 docker image tag dash-prod-gt-eg:7.0 pepsomest.azurecr.io/dash-gt-eg:7.0
 docker push pepsomest.azurecr.io/dash-gt-eg:7.0