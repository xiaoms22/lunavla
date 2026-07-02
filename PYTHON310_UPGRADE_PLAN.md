# Python 3.10+ Upgrade Plan

**Current Version:** Python 3.9.6  
**Target Version:** Python 3.10+  
**Status:** Planning

---

## 🎯 Upgrade Objectives

### Primary Goals
1. Use modern Python features (structural pattern matching, better type hints)
2. Improve performance with Python 3.10+ optimizations
3. Access newer library versions
4. Prepare for future Python ecosystem

### Benefits
- ✅ Native support for `|` union operator (already fixed for 3.9)
- ✅ Structural pattern matching (`match`/`case`)
- ✅ Better error messages
- ✅ Performance improvements (~10-15% faster)
- ✅ Access to latest PyTorch/NumPy features

---

## 📋 Upgrade Checklist

### Phase 1: Pre-upgrade Preparation
- [x] Document current Python version (3.9.6)
- [x] List all dependencies and versions
- [x] Run full test suite on Python 3.9
- [x] Verify all validations pass (7/7)
- [ ] Create upgrade branch
- [ ] Backup current environment

### Phase 2: Environment Setup
- [ ] Install Python 3.10 or 3.11
- [ ] Create new virtual environment
- [ ] Install dependencies in new environment
- [ ] Verify PyTorch compatibility
- [ ] Verify NumPy compatibility

### Phase 3: Code Migration
- [ ] Revert Python 3.9 compatibility fixes
  - [ ] Change `Union[dict[str, Any], np.ndarray]` back to `dict[str, Any] | np.ndarray`
- [ ] Update type hints to use modern syntax
- [ ] Consider using structural pattern matching where appropriate
- [ ] Update docstrings with new type syntax

### Phase 4: Testing
- [ ] Run environment check
- [ ] Run all validation scripts (7 checks)
- [ ] Run CPU smoke test
- [ ] Run BC smoke test
- [ ] Run ACT baseline training
- [ ] Run ablation experiment
- [ ] Compare results with Python 3.9 baseline

### Phase 5: Documentation
- [ ] Update README.md with Python 3.10+ requirement
- [ ] Update installation instructions
- [ ] Document any breaking changes
- [ ] Update badges

### Phase 6: Release
- [ ] Create release notes for Python 3.10+ version
- [ ] Tag release (e.g., v1.1.0-python310)
- [ ] Update main branch
- [ ] Archive Python 3.9 compatible version

---

## 🔧 Specific Changes

### 1. Type Hints Modernization

**Current (Python 3.9):**
```python
from typing import Union, Optional, Dict, List

def process_sample(sample: Union[dict[str, Any], np.ndarray]) -> Optional[Dict[str, List[float]]]:
    ...
```

**After (Python 3.10+):**
```python
def process_sample(sample: dict[str, Any] | np.ndarray) -> dict[str, list[float]] | None:
    ...
```

### 2. Pattern Matching Opportunities

**Current:**
```python
if isinstance(action, dict):
    if "pos" in action and "rot" in action:
        return process_6dof(action)
    elif "pos" in action:
        return process_3dof(action)
else:
    return process_raw(action)
```

**Potential (Python 3.10+):**
```python
match action:
    case {"pos": pos, "rot": rot}:
        return process_6dof(action)
    case {"pos": pos}:
        return process_3dof(action)
    case _:
        return process_raw(action)
```

### 3. Better Error Messages

Python 3.10+ provides more informative error messages automatically, especially for:
- Missing dictionary keys
- Type mismatches
- Import errors

---

## 📊 Expected Impact

### Performance
- **Training:** ~5-10% faster (Python 3.10 optimizations)
- **Evaluation:** ~5-10% faster
- **Data loading:** Minimal impact

### Compatibility
- **PyTorch:** Full support for 3.10+
- **NumPy:** Full support for 3.10+
- **Dependencies:** All major dependencies support 3.10+

### Code Quality
- **Type Safety:** Improved with native union types
- **Readability:** Better with modern syntax
- **Maintainability:** Easier with pattern matching

---

## 🚨 Risks and Mitigations

### Risk 1: Breaking Changes
**Risk:** Code that works on 3.9 might break on 3.10+  
**Mitigation:** Comprehensive testing before release

### Risk 2: Dependency Conflicts
**Risk:** Some dependencies might not support 3.10+ yet  
**Mitigation:** Check all dependencies before upgrading

### Risk 3: Performance Regression
**Risk:** Some operations might be slower on 3.10+  
**Mitigation:** Run performance benchmarks before and after

### Risk 4: Training Results Difference
**Risk:** Training might produce different results due to random seed differences  
**Mitigation:** Run full training comparison and document any differences

---

## 📈 Upgrade Timeline

### Option 1: Conservative (Recommended)
```
Week 1: Environment setup and dependency check
Week 2: Code migration and initial testing
Week 3: Full training comparison
Week 4: Documentation and release
```

### Option 2: Aggressive
```
Day 1-2: Environment setup
Day 3-4: Code migration
Day 5-6: Testing
Day 7: Release
```

---

## 🎯 Success Criteria

### Must Have
- [ ] All 7 validations pass
- [ ] ACT baseline achieves ≥95% success rate
- [ ] Training time ≤110% of Python 3.9 baseline
- [ ] No critical bugs

### Nice to Have
- [ ] Training time <100% of Python 3.9 baseline (faster)
- [ ] Code uses modern Python 3.10+ features
- [ ] Improved error messages in practice
- [ ] Better type checking with mypy

---

## 🔗 Resources

### Documentation
- [What's New in Python 3.10](https://docs.python.org/3/whatsnew/3.10.html)
- [What's New in Python 3.11](https://docs.python.org/3/whatsnew/3.11.html)
- [PyTorch Python 3.10+ Support](https://pytorch.org/)

### Tools
- `pyenv` for managing multiple Python versions
- `venv` for virtual environments
- `pip-compile` for dependency management

---

## 📝 Notes

### Current Python 3.9 Compatibility Fix
```python
# File: model/policy_base.py
# Current fix for Python 3.9
from typing import Union
PolicySample = Union[dict[str, Any], np.ndarray]

# Can be reverted to:
PolicySample = dict[str, Any] | np.ndarray
```

### Backward Compatibility
If we want to maintain Python 3.9 support:
- Keep current `Union` syntax
- Don't use pattern matching
- Keep supporting both versions in documentation

If we drop Python 3.9 support:
- Update to modern syntax
- Use new Python 3.10+ features
- Update README to require 3.10+

---

## 🚀 Next Steps

1. **Decide on timeline** (conservative vs aggressive)
2. **Create upgrade branch** (`feat/python310-upgrade`)
3. **Set up Python 3.10+ environment**
4. **Run initial compatibility check**
5. **Begin code migration**

---

**Created:** 2026-06-30  
**Status:** Planning  
**Owner:** Development Team  
**Priority:** Medium (Optional Enhancement)
