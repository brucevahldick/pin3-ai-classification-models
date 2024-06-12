from fastai_model import train_and_save_model, evaluate, evaluate_and_retrain_model
import sys

# total arguments
n = len(sys.argv)
print("Total arguments passed:", n)

for i in range(1, n):
    print(sys.argv[i], " ")

if sys.argv[1] == "evaluate":
    evaluate()
    print("Evaluation completed")
elif sys.argv[1] == "train":
    train_and_save_model()
    print("Model trained")
elif sys.argv[1] == "retrain":
    print(f"Retraining Model: Parameters; max_iterations={sys.argv[2]}; target_accuracy={sys.argv[3]}; "
          f"epochs={sys.argv[4]}; lr={sys.argv[5]}; momentum={sys.argv[6]}")
    evaluate_and_retrain_model(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    print("Model retrained")
