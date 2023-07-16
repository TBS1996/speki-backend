use std::{io::Write, process::Command, time::Duration};

use crate::{
    card::{generate_random_reviews, MlArgs, SavedCard},
    paths::get_ml_path,
};

pub fn log_transform(time_passed: Duration) -> f64 {
    let time_passed = time_passed.as_secs_f64();
    let time_passed_days: f64 = time_passed / (24.0 * 60.0 * 60.0); // Convert to days
    (time_passed_days + 1.0).ln() / 5.0
}

pub fn two_review_stuff() {
    // make training data
    //head_review_data(2);
    rolling_review_data(2);
    train_model("2data");
}

pub fn three_review_stuff() {
    // make training data
    rolling_review_data(3);
    train_model("3data");
}

pub fn fourplus_review_stuff() {
    // make training data
    rolling_review_data(4);
    train_model("4data");
}

pub fn five_review_stuff() {
    // make training data
    rolling_review_data(5);
    train_model("5data");
}

pub fn six_review_stuff() {
    // make training data
    rolling_review_data(6);
    train_model("6data");
}

fn train_model(model_name: &str) {
    // Specify the Python script and the training file name
    let python_script = get_ml_path().join("trainstuff.py");
    let training_file = get_ml_path().join(model_name);

    // Construct the command to execute the Python script

    let output = Command::new("python3")
        .arg(python_script)
        .arg(training_file)
        .output()
        .expect("Failed to execute command");

    let x = String::from_utf8_lossy(&output.stdout);
    dbg!(x.to_string());
}

pub fn head_review_data(len: usize) {
    let cards = SavedCard::load_all_cards();
    let mut output = String::new();

    let model_name = format!("{}data.csv", len);
    let path = get_ml_path().join(model_name);
    let _ = std::fs::remove_file(path.as_path());

    let mut points = 0;
    let mut antipoints = 0;

    for card in cards {
        let reviews = card.raw_reviews();
        let x = MlArgs::new_truncated(reviews.to_owned(), len);

        if let Some(x) = x {
            points += 1;
            output.push_str(&x.as_training_data().join(","));
            output.push('\n');
        } else {
            antipoints += 1;
        }
    }

    dbg!(points);
    dbg!(antipoints);

    // Create a new file
    let mut file = std::fs::File::create(path.as_path()).expect("Failed to create file");

    // Write the string to the file
    file.write_all(output.as_bytes())
        .expect("Failed to write to file");
}

pub fn rolling_review_data(len: usize) {
    let cards = SavedCard::load_all_cards();
    let mut output = String::new();

    let name = format!("{}data.csv", len);

    let path = get_ml_path().join(name);
    let _ = std::fs::remove_file(path.as_path());
    let mut points = 0;

    for card in cards {
        let reviews = card.raw_reviews().to_owned();

        if reviews.is_empty() {
            continue;
        }

        let all = MlArgs::new(&reviews).get_windowed(len);

        for x in all {
            points += 1;
            output.push_str(&x.as_training_data().join(","));
            output.push('\n');
        }
    }
    dbg!(points);

    // Create a new file
    let mut file = std::fs::File::create(path.as_path()).expect("Failed to create file");

    // Write the string to the file
    file.write_all(output.as_bytes())
        .expect("Failed to write to file");
}

use cpython::{ObjectProtocol, Python};

use cpython::{PyDict, PyResult};

pub fn make_prediction(model_name: &str, input_values: Vec<String>) -> PyResult<f32> {
    let model_name = format!("{}{}", get_ml_path().to_string_lossy(), model_name);
    let gil = Python::acquire_gil();
    let py = gil.python();
    py.run("import sys", None, None)?;
    py.run("import pickle", None, None)?;
    py.run("import pandas", None, None)?;
    py.run("import numpy", None, None)?;

    let code = r#"
import sys 
import pandas
import numpy
import pickle

def make_prediction(model_name, input_values):
    # Load the model from the file.
    with open(model_name + ".pkl", 'rb') as f:
        model = pickle.load(f)

    # The input values from the command line are a list. We need to convert this to a 2D numpy array.
    new_data = numpy.array([input_values])

    # Convert the numpy array to a DataFrame.
    new_data_df = pandas.DataFrame(new_data)

    # Make a prediction for the new data.
    prediction_proba = model.predict_proba(new_data_df)

    # Return the predicted probabilities.
    return prediction_proba[:, 1][0]
"#;

    let locals = PyDict::new(py);
    py.run(code, None, Some(&locals)).unwrap();

    let make_prediction = locals.get_item(py, "make_prediction").unwrap();

    let prediction: f32 = make_prediction
        .call(py, (model_name, &input_values), None)
        .unwrap()
        .extract(py)
        .unwrap();

    Ok(prediction)
}
