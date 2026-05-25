# Vehicle Cybersecurity: Intrusion Detection, CAN Bus Security, V2X Security, and ISO 21434

## Abstract

As vehicles become increasingly connected and autonomous, they also become more vulnerable to cyberattacks that could compromise safety, privacy, and functionality. This research summary provides a comprehensive examination of vehicle cybersecurity, covering threats to in-vehicle networks (particularly the CAN bus), intrusion detection systems, V2X communication security, and the ISO 21434 automotive cybersecurity standard. We analyze attack vectors from remote exploitation to physical tampering, examine defense-in-depth strategies, and discuss the unique challenges of securing safety-critical real-time systems. The automotive industry's shift toward software-defined vehicles, over-the-air updates, and vehicle-to-everything communication expands the attack surface dramatically, making cybersecurity a first-class concern for the Autonomous Vehicle Control System (AVCS). This document synthesizes current research, identifies critical open problems, and outlines future directions for securing autonomous vehicles against evolving cyber threats.

## Key Concepts

### Threat Landscape for Autonomous Vehicles
Autonomous vehicles face diverse cyber threats:
- **Remote attacks**: Exploiting wireless interfaces (WiFi, Bluetooth, cellular, V2X)
- **Short-range attacks**: Exploiting TPMS, key fob, and NFC interfaces
- **Physical attacks**: Direct access to OBD-II port, CAN bus, or vehicle ECUs
- **Supply chain attacks**: Compromised components or software during manufacturing
- **Social engineering**: Manipulating drivers or operators into compromising security
- **Adversarial AI**: Crafted inputs designed to deceive perception systems

### Controller Area Network (CAN) Bus Security
The CAN bus is the backbone of in-vehicle communication:
- **Protocol design**: Broadcast bus with no built-in authentication or encryption
- **Message format**: Arbitration ID, data payload, CRC — but no sender authentication
- **Vulnerabilities**: Any ECU can send messages as any other ECU (spoofing)
- **Denial of service**: Dominating the bus with high-priority messages
- **Fuzzing attacks**: Sending random messages to discover undocumented functionality
- **Replay attacks**: Recording and replaying legitimate messages

### Intrusion Detection Systems (IDS)
Vehicle IDS monitor for anomalous behavior:
- **Network-based IDS**: Monitoring CAN bus traffic for anomalous messages
- **Host-based IDS**: Monitoring ECU behavior for anomalous processes
- **Signature-based detection**: Matching known attack patterns
- **Anomaly-based detection**: Detecting deviations from normal behavior
- **Specification-based detection**: Detecting violations of protocol specifications

### V2X Communication Security
Securing vehicle-to-everything communication:
- **V2V (Vehicle-to-Vehicle)**: Direct vehicle-to-vehicle safety messages
- **V2I (Vehicle-to-Infrastructure)**: Traffic signal and road information
- **V2P (Vehicle-to-Pedestrian)**: Pedestrian safety messages
- **V2N (Vehicle-to-Network)**: Cloud services and internet connectivity
- **Security credential management**: SCMS (Security Credential Management System)
- **Privacy preservation**: Pseudonym certificates to prevent vehicle tracking

### ISO 21434 Automotive Cybersecurity
ISO 21434 provides a framework for automotive cybersecurity:
- **Risk assessment**: Identifying and evaluating cybersecurity risks
- **Security by design**: Integrating cybersecurity throughout the development lifecycle
- **Threat analysis and risk assessment (TARA)**: Systematic threat evaluation
- **Security validation**: Verifying security requirements are met
- **Incident response**: Procedures for responding to cybersecurity incidents
- **Continuous monitoring**: Ongoing vulnerability assessment and threat monitoring

### Defense-in-Depth Strategy
Layered security approach for vehicles:
- **Network segmentation**: Isolating safety-critical from infotainment networks
- **Firewalls and gateways**: Controlling traffic between network segments
- **Secure boot**: Verifying software integrity at startup
- **Secure update**: Cryptographically verified over-the-air updates
- **Runtime protection**: Detecting and preventing runtime attacks
- **Incident response**: Graceful degradation and recovery procedures

## State of the Art

### CAN Bus Security Enhancements
Modern approaches to CAN bus security:
- **CAN Message Authentication**: Adding message authentication codes (MACs)
  - CANauth: Lightweight authentication for CAN
  - vCAN: Virtual CAN with authentication
  - CaCAN: Centralized authentication for CAN
- **CAN FD and CAN XL**: Newer protocols with improved security features
- **CAN encryption**: Lightweight encryption for CAN payloads
- **Intrusion detection**: ML-based anomaly detection on CAN traffic
  - Deep learning: LSTM, CNN, autoencoder-based detection
  - Statistical methods: Entropy-based, frequency-based detection
  - Timed automata: Formal specification-based detection

### Automotive Ethernet Security
Transition from CAN to Automotive Ethernet:
- **IPsec and TLS**: Standard network security protocols
- **MACsec**: Layer 2 encryption for Ethernet frames
- **Some/IP**: Service-oriented middleware with security features
- **TSN (Time-Sensitive Networking)**: Deterministic Ethernet with security considerations
- **Network segmentation**: VLAN-based isolation of safety and non-safety traffic

### Secure Over-the-Air (OTA) Updates
Critical for maintaining vehicle security:
- **Code signing**: Cryptographic verification of update authenticity
- **A/B partitioning**: Atomic updates with rollback capability
- **Delta updates**: Minimizing update size for bandwidth efficiency
- **Staged rollout**: Gradual deployment to detect issues early
- **Compliance checking**: Verifying update compatibility before installation

### V2X Security Standards
Standards for secure V2X communication:
- **IEEE 1609.2**: Security services for DSRC/WAVE
- **ETSI ITS Security**: European V2X security standards
- **SCMS (Security Credential Management System)**: PKI for V2X
- **C-V2X security**: Security for cellular V2X (5G-based)
- **Misbehavior detection**: Identifying malicious V2X participants

### Hardware Security Modules (HSMs)
Dedicated hardware for cryptographic operations:
- **EVITA HSM**: European project for vehicle IT security
- **SHE (Secure Hardware Extension)**: Lightweight HSM for automotive MCUs
- **TPM (Trusted Platform Module)**: Standard security chip
- **Automotive HSMs**: Dedicated chips for ECU security (NXP, Infineon, STMicro)

### Machine Learning for Vehicle IDS
AI-based intrusion detection:
- **Deep learning approaches**: LSTM, GRU for temporal CAN traffic analysis
- **Autoencoder-based detection**: Detecting anomalies as reconstruction errors
- **Graph neural networks**: Modeling CAN message relationships
- **Federated learning**: Collaborative IDS training without sharing raw data
- **Online learning**: Adapting to evolving normal behavior patterns

## Methodologies

### Threat Analysis and Risk Assessment (TARA)
Systematic cybersecurity risk evaluation:
- **Asset identification**: What needs to be protected (safety, privacy, financial)
- **Threat identification**: Who might attack and how (STRIDE, attack trees)
- **Vulnerability analysis**: Where the system is susceptible to threats
- **Risk estimation**: Likelihood × impact assessment
- **Risk treatment**: Mitigate, transfer, accept, or avoid
- **Residual risk assessment**: Evaluating remaining risk after treatment

### Penetration Testing for Vehicles
Ethical hacking to discover vulnerabilities:
- **Black-box testing**: No knowledge of internal implementation
- **White-box testing**: Full access to source code and design documents
- **Fuzzing**: Automated generation of malformed inputs
- **Reverse engineering**: Analyzing proprietary protocols and software
- **Hardware attacks**: Side-channel analysis, fault injection, JTAG exploitation

### Security by Design
Integrating security throughout the development lifecycle:
- **Requirements phase**: Security requirements derived from TARA
- **Design phase**: Secure architecture patterns and threat modeling
- **Implementation phase**: Secure coding standards and static analysis
- **Testing phase**: Security testing, penetration testing, fuzzing
- **Deployment phase**: Secure provisioning and configuration
- **Maintenance phase**: Vulnerability monitoring and patch management

### Cryptographic Protocol Design
Designing secure communication protocols for vehicles:
- **Key management**: Generation, distribution, rotation, and revocation
- **Certificate management**: PKI design for large-scale vehicle fleets
- **Lightweight cryptography**: Efficient algorithms for resource-constrained ECUs
- **Post-quantum cryptography**: Preparing for quantum computing threats
- **Group signatures**: Privacy-preserving authentication for V2X

### Incident Response and Forensics
Handling cybersecurity incidents:
- **Detection**: Real-time monitoring and alerting
- **Containment**: Isolating affected systems to prevent spread
- **Eradication**: Removing the threat from the system
- **Recovery**: Restoring normal operation
- **Post-incident analysis**: Understanding what happened and preventing recurrence
- **Digital forensics**: Preserving and analyzing evidence

## Challenges

### Legacy Protocol Constraints
CAN bus was designed without security:
- **No authentication**: Any node can send any message
- **Limited bandwidth**: Adding security fields reduces available data bandwidth
- **Real-time requirements**: Security checks must not add unacceptable latency
- **Mixed criticality**: Security and safety requirements may conflict

### Scalability
Securing large, diverse vehicle fleets:
- **Key distribution**: Managing keys across millions of vehicles
- **Certificate management**: SCMS scaling to national and global deployment
- **Firmware diversity**: Many ECU versions requiring separate security updates
- **Heterogeneous hardware**: Different security capabilities across fleet

### Safety-Security Trade-offs
Security mechanisms can affect safety:
- **Authentication latency**: Delayed messages could affect real-time control
- **False positive blocking**: Blocking legitimate messages could cause safety issues
- **Encryption overhead**: Additional computation could affect ECU timing
- **Update failures**: Failed security updates could brick safety-critical systems

### Privacy vs. Security
Balancing privacy and security requirements:
- **V2X tracking**: V2X messages could enable vehicle tracking
- **Data collection**: Security monitoring may collect sensitive data
- **Pseudonym management**: Balancing accountability and anonymity
- **Right to repair**: Security measures may restrict independent repair

### Evolving Threat Landscape
Cyber threats constantly evolve:
- **New attack vectors**: Emerging connectivity features create new entry points
- **Advanced persistent threats**: Sophisticated, targeted attacks
- **Supply chain compromises**: Attacks during manufacturing or distribution
- **AI-powered attacks**: Machine learning used to craft more effective attacks

### Regulatory Fragmentation
Different regions have different cybersecurity regulations:
- **UN R155/R156**: UN regulations for cybersecurity and software updates
- **ISO 21434**: International standard for automotive cybersecurity
- **China GB/T**: Chinese automotive cybersecurity standards
- **US NHTSA**: US cybersecurity guidance and potential regulation
- **EU Cyber Resilience Act**: European cybersecurity legislation

## Recent Advances

### Post-Quantum Cryptography for Vehicles
Preparing for quantum computing threats:
- **NIST PQC standards**: ML-KEM (Kyber), ML-DSA (Dilithium), SLH-DSA (SPHINCS+)
- **Hybrid approaches**: Combining classical and post-quantum algorithms
- **Performance evaluation**: Benchmarking PQC on automotive hardware
- **Migration planning**: Strategies for transitioning vehicle fleets to PQC

### AI-Powered Cyber Attacks and Defenses
AI is being used by both attackers and defenders:
- **Adversarial ML attacks**: Crafting inputs to evade AI-based IDS
- **AI-powered fuzzing**: Using ML to generate more effective fuzz inputs
- **Deepfake social engineering**: AI-generated voice or video for social attacks
- **AI-powered defense**: Faster threat detection and automated response

### Zero Trust Architecture for Vehicles
Applying zero trust principles to vehicle networks:
- **Never trust, always verify**: Every communication is authenticated
- **Least privilege**: ECUs have minimum necessary permissions
- **Micro-segmentation**: Fine-grained network isolation
- **Continuous monitoring**: Real-time security state assessment
- **Adaptive access**: Dynamic permission adjustment based on context

### Secure Software-Defined Vehicles
Securing the software-defined vehicle architecture:
- **Hypervisor security**: Isolation between vehicle domains
- **Container security**: Secure deployment of containerized vehicle services
- **API security**: Securing inter-service communication
- **Configurable hardware security**: Adaptable security policies for SDVs

### Blockchain for Vehicle Security
Distributed ledger technology for automotive:
- **Decentralized PKI**: Blockchain-based certificate management
- **OTA integrity**: Verifiable update provenance on blockchain
- **Fleet attestation**: Proving software integrity across a fleet
- **Data marketplace**: Secure vehicle data sharing and monetization

## Key Papers/References

1. Miller, C., & Valasek, K. (2015). "Remote Exploitation of an Unaltered Passenger Vehicle." Black Hat.
2. Koscher, K., et al. (2010). "Experimental Security Analysis of a Modern Automobile." IEEE S&P.
3. Checkoway, S., et al. (2011). "Comprehensive Experimental Analyses of Automotive Attack Surfaces." USENIX Security.
4. ISO 21434 (2021). "Road Vehicles — Cybersecurity Engineering." International Organization for Standardization.
5. UN Regulation No. 155 (2021). "Cyber Security and Cyber Security Management System."
6. Seo, E., et al. (2018). "GIDS: GAN based Intrusion Detection System for In-Vehicle Network." PIMRC.
7. Taylor, A., et al. (2016). "Anomaly Detection in Automobile Control Network Data with Neural Networks." AAAI Workshop.
8. Kang, M., et al. (2016). "Spatiotemporal Intrusion Detection in Automotive CAN Networks." ESCAR.
9. Lo, W., et al. (2018). "A Lightweight Authentication Protocol for V2X Communications." VTC.
10. Camenisch, J., et al. (2019). "Anonymous Attestation for V2X." IEEE S&P.
11. Petit, J., et al. (2015). "Remote Attacks on Automated Vehicles Sensors." Black Hat Europe.
12. Garcia, J., et al. (2023). "Post-Quantum Cryptography for Automotive Systems." ESCAR.
13. Li, H., et al. (2022). "Federated Learning for Vehicle Intrusion Detection." IEEE T-ITS.
14. Upadhyay, D., et al. (2023). "Zero Trust Architecture for Connected Vehicles." IEEE Network.
15. Zeng, Q., et al. (2022). "A Survey of Automotive Cybersecurity: Challenges and Solutions." ACM Computing Surveys.

## Future Directions

### Quantum-Safe Vehicle Security
Transitioning vehicle security infrastructure to post-quantum algorithms before quantum computers become a practical threat, with hybrid classical/PQC schemes during the transition period.

### Autonomous Vehicle Security Certification
Developing comprehensive security certification frameworks specific to autonomous vehicles, analogous to ISO 26262 for functional safety.

### AI-Driven Autonomous Defense
Self-healing security systems that can detect, isolate, and recover from cyberattacks autonomously, using AI for real-time threat response.

### Bio-Inspired Security
Applying biological immune system principles to vehicle cybersecurity: diverse defenses, adaptation to new threats, and self/non-self discrimination.

### Secure Sensing Pipeline
End-to-end security for the sensing pipeline, from raw sensor data through perception to decision-making, protecting against adversarial inputs at every stage.

### Collaborative Security
Fleet-wide collaborative security where vehicles share threat intelligence and collectively defend against attacks, using federated learning for privacy-preserving collaboration.

### Regulatory Harmonization
Working toward globally harmonized cybersecurity regulations for autonomous vehicles, reducing compliance burden and improving security standards worldwide.

## Relevance to AVCS

Cybersecurity is a critical concern for the AVCS:

1. **In-Vehicle Network Security**: The AVCS must secure its internal CAN/Ethernet communication against message injection, spoofing, and denial-of-service attacks.

2. **V2X Security**: As the AVCS participates in V2X communication, it must authenticate messages, protect privacy, and detect misbehaving participants.

3. **OTA Update Security**: The AVCS's OTA update mechanism must ensure update authenticity, integrity, and compatibility to prevent malicious software installation.

4. **Intrusion Detection**: The AVCS incorporates ML-based IDS to detect anomalous behavior in real-time across its internal networks and external communications.

5. **ISO 21434 Compliance**: The AVCS development process follows ISO 21434 to ensure cybersecurity is integrated throughout the lifecycle.

6. **Defense-in-Depth**: The AVCS employs multiple security layers—network segmentation, authentication, encryption, and monitoring—to protect against diverse attack vectors.

7. **Safety-Security Co-Design**: The AVCS's architecture co-designs safety and security mechanisms, ensuring that security measures do not compromise real-time safety requirements.

8. **Fleet Security Management**: The AVCS benefits from fleet-wide security monitoring and collaborative threat intelligence, enabling rapid response to emerging threats.
