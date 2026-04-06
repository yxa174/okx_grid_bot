#!/bin/bash
# Автоматический перезапуск бота при падении
MAX_RESTARTS=10
RESTART_COUNT=0

while true; do
    echo "[$(date)] Запуск бота (попытка $((RESTART_COUNT+1)))"
    python okx_grid_bot.py 
    
    EXIT_CODE=$?
    RESTART_COUNT=$((RESTART_COUNT+1))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Бот остановлен корректно"
        break
    fi
    
    if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
        echo "[$(date)] Достигнут лимит перезапусков ($MAX_RESTARTS). Остановка."
        break
    fi
    
    echo "[$(date)] Бот упал (код: $EXIT_CODE). Перезапуск через 30 сек..."
    sleep 30
done
