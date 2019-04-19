#pragma once

#include "DistGradHeader.h"
#include "MPIWrapper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

#define FunctionFromMatrix(Func)													\
switch (m_type)																		\
{																					\
	case MatrixElemType::FLOAT:														\
		return m_floatMatrixPtr->Func();											\
	case MatrixElemType::HALF:														\
		return m_halfMatrixPtr->Func();												\
	case MatrixElemType::DOUBLE:													\
		return m_doubleMatrixPtr->Func();											\
	default:																		\
		RuntimeError("type not support.");											\
}	

class TypedMatrixPtr
{
public:
	TypedMatrixPtr()
		: m_type(MatrixElemType::UNKNOWN)
		, m_matrixPtr(nullptr)
		, m_doubleMatrixPtr(nullptr)
		, m_floatMatrixPtr(nullptr)
		, m_halfMatrixPtr(nullptr) {}

	TypedMatrixPtr(MatrixBasePtr matrixPtr)
		: m_matrixPtr(matrixPtr)
		, m_doubleMatrixPtr(nullptr)
		, m_floatMatrixPtr(nullptr)
		, m_halfMatrixPtr(nullptr)
	{
		if (typeid(*matrixPtr) == typeid(Matrix<float>))
		{
			m_type = MatrixElemType::FLOAT;
			m_floatMatrixPtr = dynamic_pointer_cast<SingleMatrix>(matrixPtr);
		}
		else if (typeid(*matrixPtr) == typeid(Matrix<half>))
		{
			m_type = MatrixElemType::HALF;
			m_halfMatrixPtr = dynamic_pointer_cast<HalfMatrix>(matrixPtr);
		}
		else if (typeid(*matrixPtr) == typeid(Matrix<double>))
		{
			m_type = MatrixElemType::DOUBLE;
			m_doubleMatrixPtr = dynamic_pointer_cast<DoubleMatrix>(matrixPtr);
		}
		else
			m_type = MatrixElemType::UNKNOWN;
	}

	int GetDeviceId() const { return m_matrixPtr->GetDeviceId(); }

	MatrixType GetMatrixType() const { return m_matrixPtr->GetMatrixType(); }

	size_t GetNumCols() const { FunctionFromMatrix(GetNumCols); }

	size_t GetNumRows() const { FunctionFromMatrix(GetNumRows); }

	size_t GetNumElements() const { FunctionFromMatrix(GetNumElements); }

	size_t GetNumElementsInByte() const
	{
		switch (m_type)
		{
		case MatrixElemType::FLOAT:
			return m_floatMatrixPtr->GetNumElements() * sizeof(float);
		case MatrixElemType::HALF:
			return m_halfMatrixPtr->GetNumElements() * sizeof(half);
		case MatrixElemType::DOUBLE:
			return m_doubleMatrixPtr->GetNumElements() * sizeof(double);
		default:
			RuntimeError("type not support.");
		}
	}

	MatrixBasePtr GetMatrixBasePtr() const { return m_matrixPtr; }

	std::shared_ptr<SingleMatrix> GetFloatMatrixPtr() const { return m_floatMatrixPtr; }

	void ResetFloatMatrixPtr(SingleMatrix* ptr) { m_floatMatrixPtr.reset(ptr); }

	std::shared_ptr<DoubleMatrix> GetDoubleMatrixPtr() const { return m_doubleMatrixPtr; }

	void ResetDoubleMatrixPtr(DoubleMatrix* ptr) { m_doubleMatrixPtr.reset(ptr); }

	std::shared_ptr<HalfMatrix> GetHalfMatrixPtr() const { return m_halfMatrixPtr; }

	void ResetHalfMatrixPtr(HalfMatrix* ptr) { m_halfMatrixPtr.reset(ptr); }

	MatrixElemType GetType() const { return m_type; }

	void Resize(size_t row, size_t col) const
	{
		switch (m_type)
		{
		case MatrixElemType::FLOAT:
			m_floatMatrixPtr->Resize(row, col);
			break;
		case MatrixElemType::DOUBLE:
			m_doubleMatrixPtr->Resize(row, col);
			break;
		case MatrixElemType::HALF:
			m_halfMatrixPtr->Resize(row, col);
			break;
		default:
			RuntimeError("Type is not supported.");
		}
	}

	void SetValue(double value) const
	{
		switch (m_type)
		{
		case MatrixElemType::FLOAT:
			m_floatMatrixPtr->SetValue((float)value);
			break;
		case MatrixElemType::DOUBLE:
			m_doubleMatrixPtr->SetValue(value);
			break;
		case MatrixElemType::HALF:
			m_halfMatrixPtr->SetValue((half)value);
			break;
		default:
			RuntimeError("Type is not supported.");
		}
	}

	bool operator==(const TypedMatrixPtr& other)
	{
		return other.m_matrixPtr == m_matrixPtr && other.m_type == m_type;
	}

private:
	MatrixElemType m_type;
	MatrixBasePtr m_matrixPtr;
	std::shared_ptr<SingleMatrix> m_floatMatrixPtr;
	std::shared_ptr<DoubleMatrix> m_doubleMatrixPtr;
	std::shared_ptr<HalfMatrix> m_halfMatrixPtr;
};

#undef FunctionFromMatrix



template <class ElemType>
class IDistGradAggregator
{
public:
    IDistGradAggregator(const MPIWrapperPtr& mpi)
        : m_mpi(mpi)
    {}

    virtual ~IDistGradAggregator()
    {}

    // Returns a boolean indicating if any samples were processed
    virtual bool AggregateGradients(const std::vector<Matrix<ElemType>*>& gradients, DistGradHeader* headerCPU, bool resetState) = 0;
	virtual bool AggregateGradients(const std::vector<TypedMatrixPtr>& gradients, DistGradHeader* headerCPU, bool resetState) = 0;

    size_t NumProc()
    {
        return m_mpi->NumNodesInUse();
    }

    size_t MyRank()
    {
        return m_mpi->CurrentNodeRank();
    }

    void WaitAll()
    {
        m_mpi->WaitAll();
    }

protected:
    MPIWrapperPtr m_mpi;
};

#define UsingIDistGradAggregatorMembers           \
    \
protected:                                        \
    using IDistGradAggregator<ElemType>::m_mpi;   \
    using IDistGradAggregator<ElemType>::NumProc; \
    using IDistGradAggregator<ElemType>::MyRank
} } }

namespace std
{
	template<>
	struct hash<Microsoft::MSR::CNTK::TypedMatrixPtr>
	{
		size_t operator() (const Microsoft::MSR::CNTK::TypedMatrixPtr& ptr) const
		{
			return hash<Microsoft::MSR::CNTK::MatrixBasePtr>()(ptr.GetMatrixBasePtr());
		}
	};
}
