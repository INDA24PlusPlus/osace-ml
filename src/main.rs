use mnist::*;
use ndarray::{s, Array2};
use rand::random;



fn main() {
    let (train_data, _, _, _) = load_data();


    let nn = NeuralNetwork::new(28 * 28, 10, 2, 16);

    let first_image = train_data.slice(s![.., 0]).to_owned().into_shape((28 * 28, 1)).unwrap();

    let output = nn.network(&first_image);
    print!("Output:");
    for out in output {
        print!(" {}", out);
    }
}




struct NeuralNetwork {
    layers: Vec<Layer>
}

struct Layer {
    weights: Array2<f32>,   // must have as many columns as neurons on the previous layer and as many rows as this layer
    bias: Array2<f32>,      // must have as many rows as in weights and only one column
}

impl NeuralNetwork {
    fn new(inputs: usize, outputs: usize, hidden_layers: usize, hidden_layers_size: usize) -> Self {
        let mut layers = Vec::new();

        // first layer (hidden)
        let random_weights: Vec<f32> = (0..(hidden_layers_size * inputs)).map(|_| random::<f32>()).collect();
        let random_bias: Vec<f32> = (0..hidden_layers_size).map(|_| random::<f32>()).collect();
        layers.push(Layer {
            weights: Array2::from_shape_vec((hidden_layers_size, inputs), random_weights).unwrap(),
            bias: Array2::from_shape_vec((hidden_layers_size, 1), random_bias).unwrap()
        });

        // hidden layers
        for _ in 1..hidden_layers {
            let random_weights: Vec<f32> = (0..(hidden_layers_size * hidden_layers_size)).map(|_| random::<f32>()).collect();
            let random_bias: Vec<f32> = (0..hidden_layers_size).map(|_| random::<f32>()).collect();
            layers.push(Layer {
                weights: Array2::from_shape_vec((hidden_layers_size, hidden_layers_size), random_weights).unwrap(),
                bias: Array2::from_shape_vec((hidden_layers_size, 1), random_bias).unwrap()
            });
        }

        // output layer
        let random_weights: Vec<f32> = (0..(outputs * hidden_layers_size)).map(|_| random::<f32>()).collect();
        let random_bias: Vec<f32> = (0..outputs).map(|_| random::<f32>()).collect();
        layers.push(Layer {
            weights: Array2::from_shape_vec((outputs, hidden_layers_size), random_weights).unwrap(),
            bias: Array2::from_shape_vec((outputs, 1), random_bias).unwrap(),
        });

        Self { layers }
    }

    fn network(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = neuron_layer(&output, &layer.weights, &layer.bias);
        }
        output
    }
}

fn neuron_layer(input: &Array2<f32>, weights: &Array2<f32>, bias: &Array2<f32>) -> Array2<f32> {
    sigmoid_layer(&(weights.dot(input) + bias))
}

fn sigmoid_layer(input: &Array2<f32>) -> Array2<f32> {
    input.mapv(|x| sigmoid(x))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f32) -> f32 {
    let sig = sigmoid(x);
    sig * (1.0 - sig)
}

fn load_data() -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    // https://docs.rs/mnist/latest/mnist/
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array2::from_shape_vec((28 * 28, 50_000), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f32 / 256.0);
    //println!("{:#.1?}\n",train_data.slice(s![image_num, ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((1, 50_000), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );

    let _test_data = Array2::from_shape_vec((28 * 28, 10_000), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f32 / 256.0);

    let _test_labels: Array2<f32> = Array2::from_shape_vec((1, 10_000), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    (train_data, train_labels, _test_data, _test_labels)
}