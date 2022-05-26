from flash_gradio import FlashGradio


def flash_gradio_text_classification():
    # TODO: Maybe use mock for checkpoint
    # Worst case: have a minimal checkpoint
    checkpoint_path = "checkpoint.pt"

    # Sample run data config to test workflow
    run_dict = {
        "task": "text_classification",
        # TODO: Maybe have a smaller dataset for test?
        "url": "https://pl-flash-data.s3.amazonaws.com/imdb.zip",
        "data_config": {
            "target": "from_csv",
            "input_field": "review",
            "target_fields": "sentiment",
            "train_file": "imdb/train.csv",
            "val_file": "imdb/valid.csv",
        },
    }

    flash_gradio = FlashGradio()
    flash_gradio.run(
        run_dict["task"],
        run_dict["url"],
        run_dict["data_config"],
        checkpoint_path,
    )
    assert flash_gradio.ready
