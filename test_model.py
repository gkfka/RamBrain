from argparse import Namespace
import torch.optim as optim
from datetime import datetime
from CRF_BI import *
from data_pre import *



def make_model_output(fout, word, y_pred):
    f = open(fout, 'w', encoding='utf8')
    for i, j in enumerate(word):

        for w, t in enumerate(j):

            print("{0}/{1}".format(t, y_pred[i][w]), end=' ', file=fout)
        print("\n",file=fout)
    f.close()
    return 

if __name__ == "__main__":
    args = Namespace(
        test_corpus='./corpus/corpus_test.txt',
        model_file='./first_model/spacing_RNN_CRF.pt',
        # Model hyper parameters
        embedding_size=128,
        rnn_hidden_size=128,
        tag_size=-1,
        # Training hyper parameters
        learning_rate=0.001,
        batch_size=128,
        num_epochs=100,
        early_stopping_criteria=5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    embed_model = Word2Vec.load('first_model/embeding_model.bin')

    tagset = ["MASK", "B", "I"]
    args.tag_size = len(tagset)

    # Check CUDA
    print("Device: {}".format(args.device))

    print("vectorizing ...")
    codec = Codec(embed_model, tagset)
    vectorizer = ModuVectorizer(codec)
    dataset = ModuDataset(args.test_corpus, vectorizer)

    print("testing ....")
    model = RNNCRFTagger(
        embedding_dim=args.embedding_size,
        hidden_dim=args.rnn_hidden_size,
        output_dim=args.tag_size,
        n_layers=1,
        bidirectional=True,
        dropout=False,
        pad_lid=0,
        pad_wid=0
    )
    model.load_state_dict(torch.load(args.model_file))
    model.to(args.device)

    print("test start time :", datetime.now().time(), "\n")

    batch_generator = generate_batches(dataset, _batch_size=args.batch_size, device=args.device)
    running_loss = 0.0
    running_acc = 0.0
    model.eval()

    y_pred = []
    for batch_index, (X_test, y_test) in enumerate(batch_generator):
        # compute the output
        loss = model(X_test, y_test)
        _y_pred = model.predict(X_test, y_test)

        y_pred.extend(
            [
                codec.decode_tag(t, size=vector_len(y, 0))
                for t, y in zip(_y_pred, y_test)
            ]
        )

        # compute the accuracy
        running_loss += (loss.item() - running_loss) / (batch_index + 1)

        acc_t = compute_accuracy(_y_pred, y_test)
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        if batch_index % 100 == 1:
            print(batch_index, running_loss, running_acc)

    print("Test loss: {}".format(running_loss))
    print("Test Accuracy: {}".format(running_acc))

    word_sents, _ = read_sent(args.test_corpus)
    make_model_output("pred.txt", word_sents, y_pred)


