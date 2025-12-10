# app/services/cluster_labeling.py

def label_cluster_from_summary(summary: dict):
    """
    summary: dict chứa mean của các feature cho cụm, keys include:
      'searched', 'in_cart', 'purchased', 'qty', 'total_price'
    Trả về: (name, description)
    """
    # đảm bảo có các key
    s = float(summary.get("searched", 0))
    c = float(summary.get("in_cart", 0))
    p = float(summary.get("purchased", 0))
    q = float(summary.get("qty", 0))
    total = float(summary.get("total_price", 0))

    # Loại đơn giản: các ngưỡng có thể điều chỉnh
    # 1) VIP / chi tiêu cao
    if p >= 10 or total >= 3_000_000:
        name = "Khách hàng trung thành / Chi tiêu cao (VIP)"
        desc = f"Purchased trung bình {p:.2f}, chi tiêu trung bình {total:,.0f}. Hành vi: searched={s:.1f}, in_cart={c:.1f}."
        return name, desc

    # 2) Người mua thường xuyên / tầm trung
    if p >= 5 or (q >= 8 and total >= 1_000_000):
        name = "Người mua thường xuyên / Tầm trung"
        desc = f"Mua thường xuyên: purchased≈{p:.2f}, qty≈{q:.2f}, chi tiêu≈{total:,.0f}."
        return name, desc

    # 3) Người mua tiềm năng / mới
    if 2 <= p < 5:
        name = "Người mua tiềm năng / Mới"
        desc = f"Mua trung bình {p:.2f} lần — có dấu hiệu chuyển đổi."
        return name, desc

    # 4) Người quan tâm / mua thấp: tìm kiếm cao nhưng mua thấp
    if s >= 20 and p < 3:
        name = "Người quan tâm / Mua thấp"
        desc = f"Tìm kiếm nhiều (searched≈{s:.1f}) nhưng mua thấp (purchased≈{p:.2f})."
        return name, desc

    # 5) Chỉ xem / không mua
    if p < 1 and total < 200_000:
        name = "Chỉ xem / Không mua"
        desc = f"Hành vi yếu: searched≈{s:.1f}, purchased≈{p:.2f}, chi tiêu thấp."
        return name, desc

    # Default
    name = "Nhóm khác"
    desc = f"Giá trị trung bình: searched={s:.1f}, in_cart={c:.1f}, purchased={p:.2f}, qty={q:.1f}, total_price={total:,.0f}."
    return name, desc
