
package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"text/template"
	"time"
)

type Feature struct {
	Name    string `json:"name"`
	Type    string `json:"type"`	
	Encoding string `json:"encoding,omitempty"`
}

type Target struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

type Schema struct {
	Features []Feature `json:"features"`
	Target   Target    `json:"target"`
}

type ProjectConfig struct {
	ProjectName string
	Schema      Schema
	Timestamp   string
}

const ( 
	pythonPipelineTemplate = `
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- Configuration ---
DATA_PATH = 'data.csv'
MODEL_PATH = 'model.joblib'

# --- Data Loading ---
def load_data(path):
    df = pd.read_csv(path)
    return df

# --- Preprocessing ---
def create_preprocessor(schema):
    numeric_features = [f.Name for f in schema.Features if f.Type == "int" or f.Type == "float"]
    categorical_features = [f.Name for f in schema.Features if f.Type == "string"]

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

# --- Main Pipeline ---
def run_pipeline(schema):
    print("Loading data...")
    df = load_data(DATA_PATH)

    X = df.drop(columns=[schema.Target.Name])
    y = df[schema.Target.Name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = create_preprocessor(schema)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear'))
    ])

    print("Training model...")
    model_pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Saving model...")
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print("Pipeline execution complete.")

if __name__ == '__main__':
    # This part would typically load the schema from a file or config
    # For demonstration, we use a hardcoded schema based on the example in README
    example_schema = Schema(
        Features=[
            Feature{Name: "age", Type: "int"},
            Feature{Name: "salary", Type: "float"},
            Feature{Name: "city", Type: "string", Encoding: "one-hot"}
        ],
        Target=Target{Name: "churn", Type: "bool"}
    )
    run_pipeline(example_schema)
`
	dockerfileTemplate = `
# Use a lightweight Python image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .
COPY pipeline.py .
COPY data.csv .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port if your application is a web service
# EXPOSE 8000

# Run the pipeline when the container launches
CMD ["python", "pipeline.py"]
`
	requirementsTemplate = `
pandas
scikit-learn
joblib
`
)

func main() {
	if len(os.Args) < 3 || os.Args[1] != "new" {
		fmt.Println("Usage: auto-ml-pipeline-gen new <project_name> --schema_file=<path_to_schema.json>")
		os.Exit(1)
	}

	projectName := os.Args[2]
	schemaFilePath := ""
	for _, arg := range os.Args[3:] {
		if strings.HasPrefix(arg, "--schema_file=") {
			schemaFilePath = strings.TrimPrefix(arg, "--schema_file=")
			break
		}
	}

	if schemaFilePath == "" {
		log.Fatal("Error: --schema_file argument is required.")
	}

	schema, err := loadSchema(schemaFilePath)
	if err != nil {
		log.Fatalf("Failed to load schema: %v", err)
	}

	projectPath := filepath.Join(".", projectName)
	err = os.MkdirAll(projectPath, 0755)
	if err != nil {
		log.Fatalf("Failed to create project directory: %v", err)
	}

	config := ProjectConfig{
		ProjectName: projectName,
		Schema:      schema,
		Timestamp:   time.Now().Format(time.RFC3339),
	}

	// Generate pipeline.py
	err = generateFile(filepath.Join(projectPath, "pipeline.py"), pythonPipelineTemplate, config)
	if err != nil {
		log.Fatalf("Failed to generate pipeline.py: %v", err)
	}

	// Generate Dockerfile
	err = generateFile(filepath.Join(projectPath, "Dockerfile"), dockerfileTemplate, config)
	if err != nil {
		log.Fatalf("Failed to generate Dockerfile: %v", err)
	}

	// Generate requirements.txt
	err = generateFile(filepath.Join(projectPath, "requirements.txt"), requirementsTemplate, config)
	if err != nil {
		log.Fatalf("Failed to generate requirements.txt: %v", err)
	}

	fmt.Printf("Successfully generated ML pipeline project '%s' in directory '%s'\n", projectName, projectPath)
}

func loadSchema(path string) (Schema, error) {
	var schema Schema
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return schema, fmt.Errorf("reading schema file: %w", err)
	}

	err = json.Unmarshal(data, &schema)
	if err != nil {
		return schema, fmt.Errorf("unmarshaling schema JSON: %w", err)
	}
	return schema, nil
}

func generateFile(outputPath, tmplContent string, config ProjectConfig) error {
	tmpl, err := template.New(filepath.Base(outputPath)).Parse(tmplContent)
	if err != nil {
		return fmt.Errorf("parsing template: %w", err)
	}

	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("creating output file: %w", err)
	}
	defer file.Close()

	err = tmpl.Execute(file, config)
	if err != nil {
		return fmt.Errorf("executing template: %w", err)
	}
	return nil
}

`))
