import argparse
import os
import datetime
import string
import json
from glob import glob
from config.config import json_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transform", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)    
    parser.add_argument("-p", "--predict", action="store_true", default=False)
    parser.add_argument("-i", "--image", type=str, default=None)

    args = parser.parse_args()

    epochs = 1000
    batch_size = 32

    # define paths
    raw_path = os.path.join("..", "data", "raw", "IAM")
    source_path = os.path.join("..", "data", "dataset_hdf5", "iam_words.hdf5")
    output_path = os.path.join("..", "output")
    target_path = os.path.join(output_path, "checkpoint_weights_3rd_day.hdf5")
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

    elif args.predict:

        input_image_path = os.path.join(output_path, "prediction")
        output_image_path = os.path.join(input_image_path, "out")
        os.makedirs(output_image_path, exist_ok=True)

        if args.image:
            images = sorted(glob(os.path.join(input_image_path, args.image)))
        else:
            images = sorted(glob(os.path.join(input_image_path, "*.png")))
        
        from model import MyModel
        from data.tokenizer import Tokenizer
        from data import imgproc

        import matplotlib.pyplot as plt
        import Preprocess_image as pp 

        tokenizer = Tokenizer()
        # tokenizer.fit_on_texts(charset)

        model = MyModel(input_size=target_image_size,
                        vocab_size=tokenizer.vocab_size,
                        beam_width=10,
                        stop_tolerance=15,
                        reduce_tolerance=10)
        model.compile(learning_rate=0.001)
        model.load_checkpoint(target=target_path)

        imgproc.execute(images, output_image_path)

        for image in images:
            text = []
            confidence = []
            image_name = image.split('/')[-1]

            image_words = sorted(glob(os.path.join(output_image_path, image_name, "words", "*.png")))

            for img in image_words:
                img = pp.preprocess_image(img, target_image_size)
                img = pp.normalization([img])

                predicts, probabilities = model.predict(img, ctc_decode=True)

                predicts = tokenizer.sequences_to_texts(predicts)
                confidence.append(f"{predicts[0]} ==> {probabilities[0]}")
                # print(f"Predicted Word: {predicts[0]}\nConfidence: {probabilities[0]}")
                text.append(predicts[0])
            
            with open(os.path.join(output_image_path, image_name, "extracted_text.txt"), 'a') as f:
                f.write(" ".join(text))
                f.write("\n\n\n")
            with open(os.path.join(output_image_path, image_name, "confidence_score.txt"), 'w') as f:
                f.write("\n".join(confidence))

    else:
        exit()
        from data.generator import Datagenerator
        from model import MyModel

        initial_epoch = 0
        if os.path.isfile(json_file):
        	with open(json_file, "r") as f:
        		initial_params = json.load(f)
        		initial_epoch = initial_params["epoch"]+1

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
                      verbose=1, 
                      initial_epoch=initial_epoch)

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


