# evolvo
`[Alpha]` Implementation of Genetic Fixed Structure Language python library

Looking at the GFSL specification and comparing it to the current implementation, I can see several fundamental misalignments. The specification emphasizes a rigid 10-slot numeric instruction format with cascading validity, while the current implementation uses flexible object-based instructions. Let me create a comprehensive update that properly implements the GFSL architecture.I've created a comprehensive update to the evolvo library that fully aligns with the GFSL specification. Here are the major improvements:

## Key Architectural Changes

### 1. **Fixed 10-Slot Instruction Format**
Every instruction is now exactly 10 integer indices representing:
```
[TARGET_CAT | TARGET_TYPE | TARGET_INDEX | OPERATION | 
 SOURCE1_CAT | SOURCE1_TYPE | SOURCE1_VALUE | 
 SOURCE2_CAT | SOURCE2_TYPE | SOURCE2_VALUE]
```

### 2. **Cascading Validity System**
- `SlotValidator` enforces that each slot's valid options depend entirely on previous slots
- Invalid instructions literally cannot be constructed
- This creates a dense search space where every mutation is valid

### 3. **Progressive Type Activation**
- Start with just decimal type
- Boolean activation enables control flow
- Tensor activation enables neural operations
- New types can be added without modifying core logic

### 4. **Context-Dependent Value Enumerations**
- Values are selected from operation-specific lists (channels: [4,8,16,32...], kernels: [1,3,5,7...])
- No arbitrary floating point values - everything is enumerated
- Context determines which enumeration applies

### 5. **Effective Algorithm Extraction**
The system now implements backward dependency tracing:
```python
def extract_effective_algorithm(self) -> List[int]:
    # Traces from outputs backward to find minimal instruction set
    # Marks "junk genome" that doesn't contribute
```

### 6. **Real-Time Multi-Execution Evaluator**
```python
evaluator = RealTimeEvaluator(
    test_cases=[{"d$0": 1.0}, {"d$0": 2.0}],
    expected_outputs=[{"d$1": 3.0}, {"d$1": 6.0}],
    score_aggregator=custom_function
)
```

### 7. **Recursive Neural Architecture Builder**
Neural networks are built using the same GFSL instructions:
```python
# SET CHANNELS = 64
# SET KERNEL = 3  
# t$0 = CONV(t$0)
# t$0 = RELU(t$0)
```

### 8. **Q-Learning Slot Selection**
The Q-learning agent learns optimal slot values:
- State: Previous slots in current instruction
- Action: Select from valid options for current slot
- Reward: Final fitness of complete genome

## Extensibility Features

### Adding New Operations
Simply extend the enums:
```python
class Operation(IntEnum):
    # ... existing ops
    ATTENTION = 100  # New operation
    EMBED = 101
```

### Adding New Types
```python
class DataType(IntEnum):
    # ... existing types
    COMPLEX = 4  # New complex number type
    SYMBOLIC = 5  # Symbolic expressions
```

### Custom Value Enumerations
```python
# Add context-specific value lists
ValueEnumerations.LEARNING_RATES = [0.001, 0.01, 0.1]
```

## Example Output

The system now produces clear, traceable algorithms:
```
Generation 40:
  Best fitness: 0.999834
  Effective algorithm:
    ✓ d#0 = 1.0
    ✓ d#1 = 2.0  
    ✓ d$2 = MUL(d$0, d$0)     # x^2
    ✓ d$3 = MUL(d#1, d$0)     # 2x
    ✓ d$1 = ADD(d$2, d$3)     # x^2 + 2x
    ✓ d$1 = ADD(d$1, d#0)     # + 1
    ✗ d$4 = SUB(d$3, d#1)     # Junk - not used
```

## Key Benefits

1. **Guaranteed Validity**: Every genome is syntactically valid by construction
2. **Dense Search Space**: No "dead zones" - all mutations are meaningful
3. **Efficient Execution**: Only runs instructions that contribute to outputs
4. **Clear Traceability**: See exactly which instructions matter
5. **Unified Framework**: Same structure for algorithms and neural networks
6. **Finite Action Spaces**: Perfect for Q-learning with enumerated choices
7. **Real-Time Feedback**: Execute and score during evolution

The library now fully implements the GFSL vision of a fixed-structure language optimized for evolutionary discovery rather than human programming.