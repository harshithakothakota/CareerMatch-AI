import json
import os


HISTORY_FILE = "chat_history.json"


def load_chat_history():

    if not os.path.exists(HISTORY_FILE):
        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except:

        return []


def save_chat_history(history):

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )


def add_chat_message(
    question,
    answer,
    sources
):

    history = load_chat_history()

    history.append(
        {
            "question": question,
            "answer": answer,
            "sources": sources
        }
    )

    save_chat_history(history)


def clear_chat_history():

    if os.path.exists(HISTORY_FILE):

        os.remove(HISTORY_FILE)