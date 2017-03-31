import tensorflow as tf
import utils
import sys
import os
import cPickle

num_classes=527
feature_sizes=[128]
feature_names=["audio_embedding"]
max_frames=300

def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be
      cast to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized


def get_video_matrix(features,
                      feature_size,
                      smax_frames,
                      max_quantized_value,
                      min_quantized_value):
  """Decodes features from an input string and quantizes it.

  Args:
    features: raw feature values
    feature_size: length of each frame feature vector
    max_frames: number of frames (rows) in the output feature_matrix
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    feature_matrix: matrix of all frame-features
    num_frames: number of frames in the sequence
  """
  decoded_features = tf.reshape(
      tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
      [-1, feature_size])

  num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
  feature_matrix = utils.Dequantize(decoded_features,
                                    max_quantized_value,
                                    min_quantized_value)
  feature_matrix = resize_axis(feature_matrix, 0, max_frames)
  #print feature_matrix, feature_matrix.shape, num_frames
  #sys.exit()
  return feature_matrix, num_frames


def prepare_serialized_examples(serialized_example,
    max_quantized_value=2, min_quantized_value=-2):

  contexts, features = tf.parse_single_sequence_example(
      serialized_example,
      context_features={"video_id": tf.FixedLenFeature(
          [], tf.string),
                        "labels": tf.VarLenFeature(tf.int64)},
      sequence_features={
          feature_name : tf.FixedLenSequenceFeature([], dtype=tf.string)
          for feature_name in feature_names
      })

  # read ground truth labels
  labels = (tf.cast(
      tf.sparse_to_dense(contexts["labels"].values, (num_classes,), 1,
          validate_indices=False),
      tf.bool))

  # loads (potentially) different types of features and concatenates them
  num_features = len(feature_names)
  #print num_features
  assert num_features > 0, "No feature selected: feature_names is empty!"

  assert len(feature_names) == len(feature_sizes), \
  "length of feature_names (={}) != length of feature_sizes (={})".format( \
  len(feature_names), len(feature_sizes))

  num_frames = -1  # the number of frames in the video
  feature_matrices = [None] * num_features  # an array of different features
  for feature_index in range(num_features):
    #print feature_index
    feature_matrix, num_frames_in_this_feature = get_video_matrix(
        features[feature_names[feature_index]],
        feature_sizes[feature_index],
        max_frames,
        max_quantized_value,
        min_quantized_value)
    if num_frames == -1:
      num_frames = num_frames_in_this_feature
    else:
      tf.assert_equal(num_frames, num_frames_in_this_feature)

    feature_matrices[feature_index] = feature_matrix

  # cap the number of frames at self.max_frames
  num_frames = tf.minimum(num_frames, max_frames)

  # concatenate different features
  video_matrix = tf.concat(feature_matrices, 1)

  # convert to batch format.
  video_id=contexts["video_id"]
  # TODO: Do proper batch reads to remove the IO bottleneck.
  #batch_video_ids = tf.expand_dims(contexts["video_id"], 0)
  #batch_video_matrix = tf.expand_dims(video_matrix, 0)
  #batch_labels = tf.expand_dims(labels, 0)
  #batch_frames = tf.expand_dims(num_frames, 0)

  #return batch_video_ids, batch_video_matrix, batch_labels, batch_frames
  return video_id, video_matrix, labels, num_frames

# with tf.Session() as sess:
#     filename = '/vol/vssp/msos/yx/audioset/audioset_v1_embeddings/bal_train/T4.tfrecord'
#     #filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
#     filename_queue = tf.train.string_input_producer([filename])
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     #batch_video_ids, batch_video_matrix, batch_labels, batch_frames = prepare_serialized_examples(serialized_example)
#     video_id, video_matrix, labels, num_frames = prepare_serialized_examples(serialized_example)
#     init_op = tf.initialize_all_variables()
#     sess.run(init_op)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     vid, l = sess.run([video_matrix, labels])
#     print (vid,l)
#     coord.request_stop()
#     coord.join(threads)


#if 1==1:
#    names = [ na for na in os.listdir(in_path) if na.endswith('.tfrecord') ]
#    names = sorted(names)
#def tfrd2pkl(*args):
if __name__ == '__main__':
    st=int(sys.argv[1])
    ed=int(sys.argv[2])
    print st,ed
    #for st,ed in enumerate(args):
    #  print "get st and ed parameters!\n"
    fe_fd='/vol/vssp/msos/yx/audioset/youtube8m_v2/audioset_features/eval'
    in_path='/vol/vssp/msos/yx/audioset/audioset_v1_embeddings/eval'
    names = [ na for na in os.listdir(in_path) if na.endswith('.tfrecord') ]
    for na in names[st:ed]: #Note: do not include the end index
        print "\n"
        #print na
        path = in_path + '/' + na
        with tf.Session() as sess:
            filename = path
            #filename_queue = tf.train.string_input_producer([filename], num_epochs=2)
            filename_queue = tf.train.string_input_producer([filename])
            #print filename_queue
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            #batch_video_ids, batch_video_matrix, batch_labels, batch_frames = prepare_serialized_examples(serialized_example)
            video_id, video_matrix, labels, num_frames = prepare_serialized_examples(serialized_example)
            #init_op = tf.global_variables_initializer()
            #sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)


            lists = []
            print "start to read %s:"%na 
            files=[] # same as the big tfrecord file
            while(1):            
                vid, fea, lab = sess.run([video_id, video_matrix, labels])
                if vid in lists:
                    print "Num of files in this tfrecord:%d" %len(lists) # print how many files in one tfrecord
                    out_path_fea = fe_fd + '/' + na[0:-9] + '.fea_lab_vids'
                    with open(out_path_fea,"wb") as f:
                        cPickle.dump( files, f, protocol=cPickle.HIGHEST_PROTOCOL )
                    #dict_X = cPickle.load( open( out_path_fea, 'rb' ) )
                    #for i in range(len(dict_X)):
                    #    temp=dict_X[i]
                    #    print temp['labels']
                    #    sys.exit()
                    files=[] # same as the big tfrecord file
                    lists=[]
                    break
                else:
                    print vid
                    lists.append(vid)
                    dicts={}
                    dicts['audio_embedding'] = fea
                    dicts['labels'] = lab
                    dicts['video_id'] = vid
                    files.append(dicts)                

                #dict['num_frames']=num_frames
                #print video_matrix
                #print dict
            
                #out_path_lab = fe_fd + '/' + na[0:-9] + '.lab'         
                #cPickle.dump( lab, open(out_path_lab, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
                #dict_X = cPickle.load( open( out_path_fea, 'rb' ) )
                #print dict_X['labels']
                #break;
                #print X.shape
            #print lists
            #sess.close()

            coord.request_stop()
            #coord.join(threads)
            filename_queue.dequeue()
            #reader.close()
            filename_queue.close()
            sess.run(filename_queue.close(cancel_pending_enqueues=True))
            #coord.request_stop()
            coord.join(threads, stop_grace_period_secs=5)
            sess.close()
            
            #sys.exit()
    #return 1
