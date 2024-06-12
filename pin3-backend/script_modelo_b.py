from pytorch_model import train_and_save_model, evaluate_model, retrain_model
import sys

if __name__ == "__main__":
    n = len(sys.argv)
    print("Total arguments passed:", n)

    for i in range(1, n):
        print(sys.argv[i], " ")

    if n < 2:
        print("Usage: script.py <evaluate|train|retrain> [additional arguments...]")
        sys.exit(1)

    if sys.argv[1] == "evaluate":
        evaluate_model()
        print("Evaluation completed")
    elif sys.argv[1] == "train":
        train_and_save_model()
        print("Model trained")
    elif sys.argv[1] == "retrain":
        if n < 5:
            print("Usage: script.py retrain <num_epochs> <lr> <momentum>")
            sys.exit(1)
        print(f"Retraining Model: Parameters; num_epochs={sys.argv[2]}; lr={sys.argv[3]}; momentum={sys.argv[4]}")
        retrain_model(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
        print("Model retrained")
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Usage: script.py <evaluate|train|retrain> [additional arguments...]")

