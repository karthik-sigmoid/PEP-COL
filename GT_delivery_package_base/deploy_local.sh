docker build -t dash-prod-gt-eg:3.0 -f Dockerfile.prod . 
docker run -d -p 80:80 dash-prod-gt-eg:3.0