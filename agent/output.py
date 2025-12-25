from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    """课程信息"""
    review: bool = Field(description="你帮我审核课程信息是否存在违规 存在T 不存在F")
