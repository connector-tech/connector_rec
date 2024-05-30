import asyncio
import json
from uuid import uuid4

import nltk

from datetime import datetime, timezone, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from code.db import AsyncpgConnection
from code.similarity import process_and_cluster
from loguru import logger

scheduler = AsyncIOScheduler(timezone='UTC')


async def update_user_similarity():
    users_with_questionnaire_answers = """
        SELECT user_id, json_object_agg(question_id, answer) AS answers
        FROM user_questionnaire_answers uqa
         INNER JOIN questionnaire_questions qq ON uqa.question_id = qq.id
        GROUP BY user_id;
    """
    users_with_answers = await AsyncpgConnection.fetch(users_with_questionnaire_answers)
    data_to_process = [
        {
            'user_id': user_answer['user_id'],
            **json.loads(user_answer['answers']),
        } for user_answer in users_with_answers
    ]
    logger.info(f'Processing {data_to_process} users')
    similarity_matrix = process_and_cluster(data_to_process)

    values = []
    for s in similarity_matrix:
        values.append((str(uuid4()), s['user_id'], s['user_id_2'], float(s['sim'])))

    insert_user_similarity = """
        INSERT INTO user_similarity (id, user_1_id, user_2_id, similarity)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_1_id, user_2_id) DO UPDATE SET similarity = $4;
    """

    await AsyncpgConnection.executemany(insert_user_similarity, values)


scheduler.add_job(
    update_user_similarity,
    trigger='interval',
    hours=1,
    next_run_time=datetime.now(timezone.utc) + timedelta(minutes=1),
)


async def main():
    try:
        await AsyncpgConnection.connect()

        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

        scheduler.start()

        await asyncio.Future()
    finally:
        await AsyncpgConnection.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
