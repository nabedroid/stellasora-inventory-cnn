FROM node:20-alpine

EXPOSE 3000

WORKDIR /app/

COPY ./main/package.json .
RUN npm install

COPY ./main /app/main
COPY ./common /app/common

WORKDIR /app/main

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
