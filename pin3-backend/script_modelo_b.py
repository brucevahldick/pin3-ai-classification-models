from pytorch_model import train_and_save_model, evaluate_model, retrain_model
import sys

n = len(sys.argv)
print("Total arguments passed:", n)

for i in range(1, n):
    print(sys.argv[i], " ")

if sys.argv[1] == "evaluate":
    evaluate_model()
    print("Evaluation completed")
elif sys.argv[1] == "train":
    train_and_save_model()
    print("Model trained")
elif sys.argv[1] == "retrain":
    print(f"Retraining Model: Parameters; num_epochs={sys.argv[2]}; lr={sys.argv[3]}; momentum={sys.argv[4]}")
    retrain_model(sys.argv[2], sys.argv[3], sys.argv[4])
    print("Model retrained")
