import argparse
import os
import datetime
import string

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transform", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)    
    parser.add_argument("-i", "--image", type=str)

    args = parser.parse_args()

    epochs = 1000
    batch_size = 32

    # define paths
    raw_path = os.path.join("..", "data", "raw", "IAM")
    source_path = os.path.join("..", "data", "dataset_hdf5", "iam_words.hdf5")
    output_path = os.path.join("..", "output")
    target_path = os.path.join(output_path, "checkpoint_weights.hdf5")
    os.makedirs(output_path, exist_ok = True)

    # define input size, number max of chars per line and list of valid chars
    target_image_size = (256, 64, 1)
    maxTextLength = 32
    charset = string.printable[:84]
    buf_size = 1000    

    if args.transform:

        from Create_hdf5 import Dataset

        if os.path.isfile(source_path): 
            print("Dataset file already exists")
        else:
            print("Transforming the IAM dataset..")
            ds = Dataset(raw_path)
            ds.save_partitions(source_path, target_image_size, maxTextLength)

    elif args.image:
        
        from model import MyModel
        from data.tokenizer import Tokenizer

        import matplotlib.pyplot as plt
        import Preprocess_image as pp 

        tokenizer = Tokenizer(filters = string.printable[95:], charset=charset)
        # tokenizer.fit_on_texts(charset)

        model = MyModel(input_size=target_image_size,
                        vocab_size=tokenizer.vocab_size,
                        beam_width=10,
                        stop_tolerance=15,
                        reduce_tolerance=10)
        model.compile(learning_rate=0.001)
        model.load_checkpoint(target=target_path, train=False)

        img = pp.preprocess_image(args.image, target_image_size)
        img = pp.normalization([img])

        predicts, probabilities = model.predict(img, ctc_decode=True)

        predicts = tokenizer.sequences_to_texts(predicts)
        print(f"Predicted Word: {predicts[0]}\nConfidence: {probabilities[0]}")

    else:

        from data.generator import Datagenerator
        from model import MyModel

        train_dgen = Datagenerator(source_path,
                                   charset=charset, 
                                   partition='train',
                                   batch_size=batch_size, 
                                   maxTextLength=maxTextLength, 
                                   buf_size = buf_size)
        valid_dgen = Datagenerator(source_path,
                                   charset=charset, 
                                   partition='valid',
                                   batch_size=batch_size, 
                                   maxTextLength=maxTextLength)

        print("Train_size:", train_dgen.size)
        print("Validation_size:", valid_dgen.size)

        model = MyModel(input_size=target_image_size,
                        vocab_size=train_dgen.tokenizer.vocab_size,
                        beam_width=10,
                        stop_tolerance=15,
                        reduce_tolerance=10)
        model.compile(learning_rate=0.001)
        model.summary(output_path, "summary.txt")
        model.load_checkpoint(target=target_path, train=True)
        callbacks = model.get_callbacks(logdir=output_path, checkpoint=target_path, verbose=1)

        start_time = datetime.datetime.now()

        h = model.fit(x=train_dgen,
                      epochs=epochs,
                      validation_data=valid_dgen,
                      callbacks=callbacks,
                      shuffle=False,
                      verbose=1)

        total_time = datetime.datetime.now() - start_time

        loss = h.history['loss']
        val_loss = h.history['val_loss']

        min_val_loss = min(val_loss)
        min_val_loss_i = val_loss.index(min_val_loss)

        time_epoch = (total_time / len(loss))
        total_item = (train_dgen.size + valid_dgen.size)

        t_corpus = "\n".join([
            f"Total train images:      {train_dgen.size}",
            f"Total validation images: {valid_dgen.size}",
            f"Batch:                   {train_dgen.batch_size}\n",
            f"Total time:              {total_time}",
            f"Time per epoch:          {time_epoch}",
            f"Time per item:           {time_epoch / total_item}\n",
            f"Total epochs:            {len(loss)}",
            f"Best epoch:              {min_val_loss_i + 1}\n",
            f"Best validation loss:    {min_val_loss}\n",
            f"Training loss:           {loss[min_val_loss_i]:.8f}",
            f"Validation loss:         {min_val_loss:.8f}"
        ])

        with open(os.path.join(output_path, "train.txt"), "a") as lg:
            lg.write(t_corpus)
            print(t_corpus)


