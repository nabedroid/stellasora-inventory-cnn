FROM node:20-alpine

EXPOSE 3001

WORKDIR /app
COPY ./manager/package.json ./
RUN npm install

COPY ./manager /app/manager
COPY ./common /app/common

WORKDIR /app/manager

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
