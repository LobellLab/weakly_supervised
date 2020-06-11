import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_cam(sess, cam, dense_kernel, dense_bias, predictions, model_spec, model_dir, metrics_val, epoch, num_steps):
    cam_dir = os.path.join(model_dir, 'eval_cam')
    if not os.path.exists(cam_dir):
        os.makedirs(cam_dir)
    best_acc = 0.0
    best_acc_dir = os.path.join(cam_dir, 'best_eval_acc.npy')
    if os.path.isfile(best_acc_dir):
        best_acc = np.load(best_acc_dir)[0]

    if metrics_val['accuracy'] > best_acc:
        print("Saving CAM for best epoch so far...")
        np.save(best_acc_dir, np.array([metrics_val['accuracy'], epoch]))
        sess.run(model_spec['iterator_init_op'])
        for i in range(num_steps):
            cam_val, dense_kernel_val, dense_bias_val, predictions_val = sess.run([cam, dense_kernel, dense_bias, predictions])
            np.save(os.path.join(cam_dir, 'eval-cam-batch'+str(i)), cam_val)
            np.save(os.path.join(cam_dir, 'eval-dense-kernel-batch'+str(i)), dense_kernel_val)
            np.save(os.path.join(cam_dir, 'eval-dense-bias-batch'+str(i)), dense_bias_val)
            np.save(os.path.join(cam_dir, 'eval-prediction-batch'+str(i)), predictions_val)


def get_best_threshold(sess, model_spec, num_steps, params):

    sample_steps = params.threshold_sample_steps
    batch_size = params.batch_size
    train_size = params.train_size
    if train_size > 20000:
        CAMs = np.zeros((int(batch_size * (num_steps // sample_steps + 1)), params.CAM_size, params.CAM_size))
        labels_vals = np.zeros((int(batch_size * (num_steps // sample_steps + 1)), 1))
    else:
        CAMs = np.zeros((train_size, params.CAM_size, params.CAM_size))
        labels_vals = np.zeros((train_size, 1))
        sample_steps = 1

    CAM_input = model_spec['CAM_input']
    dense_kernel = model_spec['dense_kernel']
    dense_bias = model_spec['dense_bias']
    
    loss = model_spec['loss']
    labels = model_spec['labels']
    
    sess.run(model_spec['iterator_init_op'])
   
    logging.info("- Computing optimal threshold on training set...") 
    t = trange(num_steps)
    j = 0
    
    for i in t:
        if i % sample_steps == 0:
            CAM_input_val, dense_kernel_val, dense_bias_val, labels_val = sess.run([CAM_input, dense_kernel, dense_bias, labels])
            if CAM_input_val.shape[0] != batch_size:
                continue
            
            CAM = np.sum(np.matmul(CAM_input_val, dense_kernel_val) + dense_bias_val, axis=-1)
            CAMs[j*batch_size:(j+1)*batch_size,:,:] = np.array(CAM)
            labels_vals[j*batch_size:(j+1)*batch_size,:] = np.array(labels_val)
           
            j = j + 1
            print("dense kernel shape:", dense_kernel_val.shape)
    
    def get_classif_acc(threshold, cam, ground_truth):
        segpreds = cam > threshold # all cam values > threshold become 1
        preds = np.mean(segpreds, axis=(1,2)) > 0.5 # image-level label becomes 1 if >0.5 pixels are 1
        acc = sum(preds == ground_truth) / len(ground_truth)
        return acc

    possible_thresholds = list(np.linspace(-10,10,1001))
    labels_vals = labels_vals.flatten().astype(bool)
    classif_accs = [get_classif_acc(x, CAMs, labels_vals) for x in possible_thresholds]
    
    return np.array(possible_thresholds)[np.argmax(np.array(classif_accs))], np.max(classif_accs)


def get_seg_metrics(sess, model_spec, num_steps, params, threshold):
    
    CAM_input = model_spec['CAM_input']
    dense_kernel = model_spec['dense_kernel']
    dense_bias = model_spec['dense_bias']
    seg_labels = model_spec['seg_labels']
    
    acc = 0.0
    pre = 0.0
    rec = 0.0
    f1s = 0.0

    sess.run(model_spec['iterator_init_op'])
    
    print("- Computing segmentation accuracy...")
    t = trange(num_steps)

    for i in t:
        CAM_input_val, dense_kernel_val, dense_bias_val, seg_values = sess.run([CAM_input, dense_kernel, dense_bias, seg_labels])
        CAM = np.sum(np.matmul(CAM_input_val, dense_kernel_val) + dense_bias_val, axis=-1)
        seg_pred = CAM > threshold
        offset = (seg_values.shape[1] - params.CAM_size) // 2
        seg_true = seg_values[:,offset:params.CAM_size+offset,offset:params.CAM_size+offset]
        seg_true = seg_true.astype(int)
        acc += accuracy_score(seg_true.flatten(), seg_pred.flatten()) 
        pre += precision_score(seg_true.flatten(), seg_pred.flatten())
        rec += recall_score(seg_true.flatten(), seg_pred.flatten())
        f1s += f1_score(seg_true.flatten(), seg_pred.flatten())
    
    return acc / num_steps, pre / num_steps, rec / num_steps, f1s / num_steps


def train_sess(sess, model_spec, num_steps, writer, params, model_dir, epoch):

    # Get relevant graph operations or nodes needed for training
    loss = model_spec['loss']
    train_op = model_spec['train_op']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()

    # Load training dataset into pipeline and initialize the metrics local variables
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    # Use tqdm for progress bar
    t = trange(num_steps)
    for i in t:
        handles = [train_op, update_metrics, loss]
        # Evaluate summaries every 100 steps
        if i % 100 == 0:
            _, _, loss_val, summ, global_step_val = sess.run(
                handles + [summary_op, global_step])

            # Write training summary to tensorboard
            writer.add_summary(summ, global_step_val)
        else:
            _, _, loss_val = sess.run(handles)

    # Log the loss in the tqdm progress bar
    t.set_postfix(loss='{:05.3f}'.format(loss_val))

    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)

    return metrics_val


def evaluate_sess(sess, model_spec, num_steps, writer, params, model_dir, epoch):
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()
    predictions = model_spec['predictions']
    if params.model == 'UNet':
        CAM_input = model_spec['CAM_input']
        dense_kernel = model_spec['dense_kernel']
        dense_bias = model_spec['dense_bias']

    # Load evaluation dataset into pipeline and initialize the metrics local variables
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    for i in range(num_steps):
        handles = [update_metrics]
        if i % 100 == 0:
            handles += [summary_op, global_step]
            _, summ, global_step_val = sess.run(handles)
            writer.add_summary(summ, global_step_val)
        else:
            # adds results automatically to metrics
            sess.run(handles)

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)

    if params.model == 'UNet' and params.save_cam and epoch > 5:
        save_cam(sess, CAM_input, dense_kernel, dense_bias, predictions, model_spec, model_dir, metrics_val, epoch, num_steps)

    return metrics_val


def train_and_evaluate(train_model_spec, eval_model_spec, test_model_spec, model_dir, params, restore_from=None, ts_uuid=''):
    model_dir = Path(model_dir)

    train_loss = []
    train_acc = []
    train_segacc_list = []
    train_segpre_list = []
    train_segrec_list = []
    train_segf1s_list = []

    eval_loss = []
    eval_acc = []
    eval_segacc_list = []
    eval_segpre_list = []
    eval_segrec_list = []
    eval_segf1s_list = []

    test_loss = []
    test_acc = []
    test_segacc_list = []
    test_segpre_list = []
    test_segrec_list = []
    test_segf1s_list = []

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize model variables
        sess.run(train_model_spec['variable_init_op'])

        # Set up tensorboard files
        train_summary_dir = model_dir / f'{ts_uuid}_train_summaries'
        eval_summary_dir = model_dir / f'{ts_uuid}_eval_summaries'
        test_summary_dir = model_dir / f'{ts_uuid}_test_summaries'
        train_summary_dir.mkdir(exist_ok=True)
        eval_summary_dir.mkdir(exist_ok=True)
        test_summary_dir.mkdir(exist_ok=True)
        train_writer = tf.summary.FileWriter(str(train_summary_dir), sess.graph)
        eval_writer = tf.summary.FileWriter(str(eval_summary_dir), sess.graph)
        test_writer = tf.summary.FileWriter(str(test_summary_dir), sess.graph)

        best_accuracy = 0.0

        for epoch in range(params.num_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

            # Compute number of batches in one epoch
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
            train_metrics = train_sess(sess, train_model_spec, num_steps, train_writer, params, model_dir, epoch)

            # if gap_sigmoid model, find best threshold
            if params.unet_type == 'gap_sigmoid':
                best_thresh, best_acc = get_best_threshold(sess, train_model_spec, num_steps, params)
                logging.info('- Best threshold on train samples: {:0.4f}, best accuracy: {:0.4f}'.format(best_thresh, best_acc)) 

                train_segacc, train_segpre, train_segrec, train_segf1s = get_seg_metrics(sess, train_model_spec, num_steps, params, best_thresh)
                train_segacc_list.append(train_segacc)
                train_segpre_list.append(train_segpre)
                train_segrec_list.append(train_segrec)
                train_segf1s_list.append(train_segf1s)
                logging.info('- Train segmentation accuracy: {:0.4f}'.format(train_segacc))
            else:
                train_segacc_list.append(train_metrics['segmentation_acc'])

            # Evaluate for one epoch on validation set
            num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
            eval_metrics = evaluate_sess(sess, eval_model_spec, num_steps, eval_writer, params, model_dir, epoch)

            if params.unet_type == 'gap_sigmoid':
                eval_segacc, eval_segpre, eval_segrec, eval_segf1s = get_seg_metrics(sess, eval_model_spec, num_steps, params, best_thresh)
                eval_segacc_list.append(eval_segacc)
                eval_segpre_list.append(eval_segpre)
                eval_segrec_list.append(eval_segrec)
                eval_segf1s_list.append(eval_segf1s)
                logging.info('- Eval segmentation accuracy: {:0.4f}'.format(eval_segacc))
            else:
                eval_segacc.append(eval_metrics['segmentation_acc'])

            # Evaluate for one epoch on test set
            num_steps = (params.test_size + params.batch_size - 1) // params.batch_size
            test_metrics = evaluate_sess(sess, test_model_spec, num_steps, test_writer, params, model_dir, epoch)

            if params.unet_type == 'gap_sigmoid':
                test_segacc, test_segpre, test_segrec, test_segf1s = get_seg_metrics(sess, test_model_spec, num_steps, params, best_thresh)
                test_segacc_list.append(test_segacc)
                test_segpre_list.append(test_segpre)
                test_segrec_list.append(test_segrec)
                test_segf1s_list.append(test_segf1s)
                logging.info('- Test segmentation accuracy: {:0.4f}'.format(test_segacc))
            else:
                test_segacc.append(test_metrics['segmentation_acc'])

            train_loss.append(train_metrics['loss'])
            train_acc.append(train_metrics['accuracy'])
            eval_loss.append(eval_metrics['loss'])
            eval_acc.append(eval_metrics['accuracy'])
            test_loss.append(test_metrics['loss'])
            test_acc.append(test_metrics['accuracy'])

            # Save best model so far based on task accuracy (image classification)
            if eval_metrics['accuracy'] > best_accuracy:
                print("Saving best model...")
                saver.save(sess, os.path.join(model_dir, "best_model.ckpt"))
            best_accuracy = max(best_accuracy, eval_metrics['accuracy'])

    # Write metrics to disk for easy analysis
    df = pd.DataFrame({'train_loss': train_loss, 'eval_loss': eval_loss, 'test_loss': test_loss,
                       'train_accuracy': train_acc, 'eval_accuracy': eval_acc, 'test_accuracy': test_acc,
                       'train_segacc': train_segacc_list, 'eval_segacc': eval_segacc_list, 'test_segacc': test_segacc_list,
                       'train_segpre': train_segpre_list, 'eval_segpre': eval_segpre_list, 'test_segpre': test_segpre_list,
                       'train_segrec': train_segrec_list, 'eval_segrec': eval_segrec_list, 'test_segrec': test_segrec_list,
                       'train_segf1s': train_segf1s_list, 'eval_segf1s': eval_segf1s_list, 'test_segf1s': test_segf1s_list})
    df.to_csv(os.path.join(model_dir, 'metrics.csv'), index=False)


