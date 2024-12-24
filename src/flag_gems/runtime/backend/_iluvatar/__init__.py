from backend_utils import Autograd, VendorInfoBase  # noqa: E402

from .ops import *  # noqa: F403

vendor_info = VendorInfoBase(
    vendor_name="iluvatar", device_name="cuda", device_query_cmd="ixsmi"
)


def get_register_op_config():
    return ()


def get_unused_op():
    return ()


__all__ = ["*"]
