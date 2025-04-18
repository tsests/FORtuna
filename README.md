# Документация по запуску бэкенда проекта

## Работа тестера

URL: https://docs.google.com/spreadsheets/d/1j8V2_UTXMmGGIpf62Esr9_Q_wEZ6Hq0TgQrx8Uu1n8k/edit?gid=331233788#gid=331233788

## Работа аналитика

URL: https://docs.google.com/document/d/1zKy_yc1iZ5UBc9J-3iD1YBQQxgk2I4pJ10IPf_I2JhQ/edit?tab=t.0

## Предварительные требования

1. Установленный Docker и Docker Compose
2. Доступ к терминалу/командной строке

## 1. Настройка окружения

Перед запуском проекта необходимо создать файл `.env` в корневой директории проекта со следующими переменными:
```
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_database_name
POSTGRES_HOST=postgres-mts
POSTGRES_PORT=5432

MINIO_ROOT_USER=your_minio_username
MINIO_ROOT_PASSWORD=your_minio_password
MINIO_ENDPOINT=minio
MINIO_BUCKET=your_bucket_name
```


## 2. Запуск сервисов

### Вариант 1: Запуск отдельных контейнеров

1. **PostgreSQL**:
   ```bash
   docker run --name postgres-mts \
     -e POSTGRES_USER=your_username \
     -e POSTGRES_PASSWORD=your_password \
     -e POSTGRES_DB=your_database_name \
     -p 5432:5432 \
     -d postgres
   ```
2. **MinIO**:
```
  docker run -p 9000:9000 -p 9001:9001 \
    --name minio \
    -e "MINIO_ROOT_USER=your_minio_username" \
    -e "MINIO_ROOT_PASSWORD=your_minio_password" \
    -d quay.io/minio/minio server /data --console-address ":9001"
```
### Вариант 2: Использование Docker Compose (рекомендуется)

Создайте файл docker-compose.yml:
```
version: '3.8'

services:
  postgres:
    image: postgres
    container_name: postgres-mts
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  minio:
    image: quay.io/minio/minio
    container_name: minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

  pgadmin:
    image: dpage/pgadmin4
    container_name: pgadmin
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@admin.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    depends_on:
      - postgres

  backend:
    build: .
    container_name: backend
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - MINIO_ROOT_USER=${MINIO_ROOT_USER}
      - MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}
      - MINIO_ENDPOINT=minio
      - MINIO_BUCKET=${MINIO_BUCKET}
    depends_on:
      - postgres
      - minio

volumes:
  postgres_data:
  minio_data:
```

Запустите сервисы командой:

```
docker-compose up -d
```

## 3. Настройка MinIO

После запуска MinIO:
1. Откройте консоль MinIO в браузере: http://localhost:9001
2. Войдите с указанными в .env учетными данными
3. Создайте новый bucket с именем, указанным в MINIO_BUCKET
4. Сделайте bucket публичным:
 - Перейдите в Bucket
 - Откройте вкладку "Access"
 - Нажмите "Edit policy"
 - Выберите "Public" и сохраните

## 4. Сборка и запуск бэкенда

Если вы используете отдельные контейнеры (Вариант 1):
Соберите образ бэкенда:
```
docker build -t backend .
```
Запустите контейнер:
```
docker run -p 8000:8000 --name backend --link postgres-mts:postgres --link minio:minio -d backend
```

## 5. Проверка работы

После успешного запуска:
Бэкенд будет доступен на http://localhost:8000
MinIO Console на http://localhost:9001
PgAdmin на http://localhost:5050

## Примечания
Убедитесь, что порты 5432, 8000, 9000, 9001 и 5050 не заняты другими сервисами
Для production окружения рекомендуется изменить дефолтные учетные данные
Все данные в контейнерах будут сохранены благодаря volumes, но для production рекомендуется настроить регулярное резервное копирование
