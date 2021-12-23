
from argparse import Namespace
import torch.optim as optim
from datetime import datetime
from BiGRU_CRF import *
from word2vec import *

if __name__ == "__main__":
    args = Namespace(
        train_corpus        = './corpus/corpus_train.txt',
        model_file          = './model/spacing_RNN_CRF',
        # Model hyper parameters
        embedding_size      = 128,
        rnn_hidden_size     = 128,
        tag_size            = -1,
        # Training hyper parameters
        learning_rate       = 0.001,
        batch_size          = 128,
        num_epochs          = 100,
        early_stopping_criteria=5,
        device              = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    embed_model = Word2Vec.load('model/embeding_model.bin')

    tagset = ["MASK", "B", "I"]
    args.tag_size = len(tagset)

    # Check CUDA
    print("Device: {}".format(args.device))

    #print("reading ...")
    #train_sents, train_tag_sents = get_bigram_sents(args.train_corpus)
    #print("corpus sentence count : ", len(train_sents))

    print("vectorizing ...")
    codec = Codec(embed_model, tagset)
    vectorizer = ModuVectorizer(codec)
    dataset = ModuDataset(args.train_corpus, vectorizer)

    print("training ....")
    model = RNNCRFTagger(
        embedding_dim = args.embedding_size,
        hidden_dim = args.rnn_hidden_size,
        output_dim = args.tag_size,
        n_layers = 1,
        bidirectional = True,
        dropout = False,
        pad_lid = 0,
        pad_wid = 0
    )
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("train start time :", datetime.now().time(), "\n")
    for epoch_index in range(args.num_epochs):
        batch_generator = generate_batches(dataset, _batch_size=args.batch_size, device=args.device)
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        epoch_index += 1
        for batch_index, (X_train, y_train) in enumerate(batch_generator):
            #print(X_train.shape, y_train.shape)

            # the training routine is these 5 steps:
            # --------------------------------------
            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            loss = model(X_train, y_train)
            y_pred = model.predict(X_train, y_train)
            
            # step 3. compute the loss

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()
            # -----------------------------------------
            # compute the  running loss and running accuracy
            running_loss += (loss.item() - running_loss) / (batch_index + 1)
            acc_t = compute_accuracy(y_pred, y_train)

            running_acc += (acc_t - running_acc) / (batch_index + 1)
            if batch_index % 10 == 1:
                print("-------------------------------------------")
                print("\nepoch ", epoch_index)
                print("batch_index ", batch_index)
                print("running_loss = {}".format(running_loss))
                print("running_acc  = {}".format(running_acc))
                print("time :", datetime.now().time())
                print("-------------------------------------------")

        print("-------------------------------------------")
        print("\nepoch ", epoch_index)
        print("running_loss = {}".format(running_loss))
        print("running_acc  = {}".format(running_acc))
        print("time :", datetime.now().time())
        print("-------------------------------------------")
        if epoch_index % 5 == 1:
            fnm = "{}_{}".format(args.model_file, epoch_index)
            torch.save(model.state_dict(), fnm+'.pt')

    print("train end time :", datetime.now().time())
    torch.save(model.state_dict(), args.model_file+'.pt')

