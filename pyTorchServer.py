import flwr as fl


if __name__ == "__main__":

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_eval=1.0,
    )

    # Start server
    fl.server.start_server(
        server_address="localhost:8080",
        config={"num_rounds": 1},
        strategy=strategy,
        grpc_max_message_length = 1122211497
    )