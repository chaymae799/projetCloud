# Smart Contract Hyperledger Fabric

Ce smart contract Go doit être déployé dans Hyperledger Fabric.

## Déploiement

Copiez ces fichiers dans `fabric-samples/chaincode/satellite-tle/` puis :
```bash
cd fabric-samples/test-network
./network.sh deployCC -ccn satellite -ccp ../chaincode/satellite-tle -ccl go -c tleschannel
```
