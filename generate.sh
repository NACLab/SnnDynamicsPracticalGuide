echo "Generating Experiment Folders"
mkdir -p "exp/inputEncoding/boundedInput/bernoulli"
mkdir -p "exp/inputEncoding/boundedInput/poisson"
mkdir -p "exp/inputEncoding/periodicEncoding/phasor"
mkdir -p "exp/inputEncoding/periodicEncoding/wave"
mkdir -p "exp/inputEncoding/sparseInput/event"
mkdir -p "exp/inputEncoding/unboundedInput/latency"

echo "Running Input Encoding"
echo "-- Running Bounded Input"
echo "---- Running Bernoulli"
python -m src.inputEncoding.boundedInput.bernoulli.main
echo "---- Running Poisson"
python -m src.inputEncoding.boundedInput.poisson.main

echo "-- Running Periodic Encoding"
echo "---- Running Phasor"
python -m src.inputEncoding.periodicEncoding.phasor.main
echo "---- Running Wave"
python -m src.inputEncoding.periodicEncoding.wave.main

echo "-- Running Sparse Input"
echo "---- Running Event"
python -m src.inputEncoding.sparseInput.event.main

echo "-- Running Unbounded Input"
echo "---- Running Latency"
python -m src.inputEncoding.unboundedInput.latency.main
