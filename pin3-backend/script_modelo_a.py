from fastai_model import train_and_save_model, evaluate, evaluate_and_retrain_model
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: script.py <evaluate|train|retrain> [additional arguments...]")
        sys.exit(1)

    if sys.argv[1] == "evaluate":
        evaluate()
        print("Evaluation completed")
    elif sys.argv[1] == "train":
        train_and_save_model()
        print("Model trained")
    elif sys.argv[1] == "retrain":
        if len(sys.argv) < 7:
            print("Usage: script.py retrain <max_iterations> <target_accuracy> <epochs> <lr> <momentum>")
            sys.exit(1)
        print(f"Retraining Model: Parameters; max_iterations={sys.argv[2]}; target_accuracy={sys.argv[3]}; "
              f"epochs={sys.argv[4]}; lr={sys.argv[5]}; batch_size={sys.argv[6]}")
        evaluate_and_retrain_model(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
        print("Model retrained")
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Usage: script.py <evaluate|train|retrain> [additional arguments...]")
