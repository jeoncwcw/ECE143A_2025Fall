from neural_decoder.neural_decoder_trainer import loadModel, getDatasetLoaders
import argparse
import torch
import numpy as np
from edit_distance import SequenceMatcher

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--modelDir", type=str, default=None, help="Path to model")
    parser.add_argument("--datasetPath", type=str, default="/mnt/data/jeon/competitionData/ptDecoder_ctc", help="Path to dataset")
    parser.add_argument("--batchSize", type=int, default=128, help="Batch size")
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    input_args = parser.parse_args()
    model = loadModel(input_args.modelDir, device=device)
    _, testLoader, _ = getDatasetLoaders(
        input_args.datasetPath,
        input_args.batchSize,
    )

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    with torch.no_grad():
        model.eval()
        allLoss = []
        total_edit_distance = 0
        total_seq_length = 0
        
        for batch_idx, (X, y, X_len, y_len, testDayIdx) in enumerate(testLoader):
            X, y, X_len, y_len, testDayIdx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                testDayIdx.to(device),
            )

            pred = model.forward(X, testDayIdx)
            loss = loss_ctc(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                y_len,
            )
            
            allLoss.append(loss.item()) 

            adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
            
           
            for iterIdx in range(pred.shape[0]):
                decodedSeq = torch.argmax(
                    pred[iterIdx, 0 : adjustedLens[iterIdx], :], # 여기서 torch.tensor()로 또 감쌀 필요 없음
                    dim=-1,
                )
                decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                decodedSeq = decodedSeq.cpu().detach().numpy()
                decodedSeq = np.array([i for i in decodedSeq if i != 0])

                trueSeq = np.array(
                    y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                )

                matcher = SequenceMatcher(
                    a=trueSeq.tolist(), b=decodedSeq.tolist()
                )
                total_edit_distance += matcher.distance()
                total_seq_length += len(trueSeq)
            
            
            current_cer = total_edit_distance / total_seq_length
            current_acc = 1 - current_cer
            
            current_loss = np.mean(allLoss) 
            
            print(f"Batch [{batch_idx+1}/{len(testLoader)}] Loss: {current_loss:.4f}, CER: {current_cer:.4f}, Accuracy: {current_acc:.4f}")

    final_cer = total_edit_distance / total_seq_length
    final_acc = 1 - final_cer
    print("\n" + "="*30)
    print(f"FINAL RESULT")
    print(f"CER: {final_cer:.4f}")
    print(f"Accuracy: {final_acc:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()