package main

import (
    "encoding/json"
    "fmt"
    "github.com/hyperledger/fabric-contract-api-go/contractapi"
)

type SmartContract struct {
    contractapi.Contract
}

type TLE struct {
    SatelliteName string `json:"satelliteName"`
    Line1         string `json:"line1"`
    Line2         string `json:"line2"`
    Timestamp     string `json:"timestamp"`
}

func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
    genesis := TLE{
        SatelliteName: "GENESIS",
        Line1:         "Genesis Block OrbitalChain",
        Line2:         "Blockchain pour satellites",
        Timestamp:     "2024-11-28",
    }
    tleJSON, _ := json.Marshal(genesis)
    return ctx.GetStub().PutState("TLE0", tleJSON)
}

func (s *SmartContract) CreateTLE(ctx contractapi.TransactionContextInterface, id string, satelliteName string, line1 string, line2 string, timestamp string) error {
    tle := TLE{
        SatelliteName: satelliteName,
        Line1:         line1,
        Line2:         line2,
        Timestamp:     timestamp,
    }
    tleJSON, _ := json.Marshal(tle)
    return ctx.GetStub().PutState(id, tleJSON)
}

func (s *SmartContract) QueryTLE(ctx contractapi.TransactionContextInterface, id string) (*TLE, error) {
    tleJSON, err := ctx.GetStub().GetState(id)
    if err != nil {
        return nil, fmt.Errorf("Failed to read: %v", err)
    }
    if tleJSON == nil {
        return nil, fmt.Errorf("TLE does not exist: %s", id)
    }
    var tle TLE
    json.Unmarshal(tleJSON, &tle)
    return &tle, nil
}

func (s *SmartContract) QueryAllTLE(ctx contractapi.TransactionContextInterface) ([]*TLE, error) {
    resultsIterator, err := ctx.GetStub().GetStateByRange("", "")
    if err != nil {
        return nil, err
    }
    defer resultsIterator.Close()
    
    var tles []*TLE
    for resultsIterator.HasNext() {
        queryResponse, _ := resultsIterator.Next()
        var tle TLE
        json.Unmarshal(queryResponse.Value, &tle)
        tles = append(tles, &tle)
    }
    return tles, nil
}

func main() {
    chaincode, _ := contractapi.NewChaincode(&SmartContract{})
    chaincode.Start()
}
