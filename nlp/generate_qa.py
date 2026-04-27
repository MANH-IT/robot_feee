# -*- coding: utf-8 -*-
"""
generate_custom_qa.py - Sinh dữ liệu câu hỏi - đáp án tùy chỉnh cho robot
Phiên bản: 2.0 - Mở rộng lên 500+ cặp QA
"""

import json
import os
import random
from typing import List, Dict

# ============================================================
# DỮ LIỆU CÂU HỎI - ĐÁP ÁN CHÍNH (MỞ RỘNG)
# ============================================================

QA_DATA: List[Dict] = [
    # ==================== HỌC TẬP (Hoc tap) ====================
    {
        "question": "học ngành cntt có khó không",
        "answer": "Ngành CNTT yêu cầu tư duy logic và khả năng tự học cao. Nếu có đam mê, bạn sẽ thấy rất thú vị. Giảng viên Khoa CNTT luôn sẵn sàng hỗ trợ!",
        "category": "hoc_tap",
        "keywords": ["cntt", "khó", "học", "ngành"]
    },
    {
        "question": "học ngành cơ khí có dễ xin việc không",
        "answer": "Ngành Cơ khí có nhu cầu nhân lực rất lớn. Sinh viên tốt nghiệp có thể làm việc tại các nhà máy, công ty sản xuất, tập đoàn ô tô, nhà máy chế tạo máy.",
        "category": "hoc_tap",
        "keywords": ["cơ khí", "việc làm", "xin việc", "tốt nghiệp"]
    },
    {
        "question": "ngành kỹ thuật xây dựng học những môn gì",
        "answer": "Ngành Kỹ thuật xây dựng bao gồm các môn: Cơ học kết cấu, Vật liệu xây dựng, Kết cấu bê tông cốt thép, Thi công công trình, Địa kỹ thuật, Thủy lực công trình.",
        "category": "hoc_tap",
        "keywords": ["xây dựng", "môn học", "kỹ thuật", "học"]
    },
    {
        "question": "ngành điện điện tử học những môn gì",
        "answer": "Ngành Điện - Điện tử gồm: Mạch điện, Điện tử cơ bản, Vi xử lý, Điều khiển tự động, Hệ thống nhúng, Truyền thông số, Xử lý tín hiệu số.",
        "category": "hoc_tap",
        "keywords": ["điện", "điện tử", "môn học", "kỹ thuật"]
    },
    {
        "question": "ngành vận tải kinh tế học những môn gì",
        "answer": "Ngành Vận tải Kinh tế gồm: Kinh tế vận tải, Quản trị logistics, Khai thác vận tải, Quản lý chuỗi cung ứng, Kinh tế đô thị, Quy hoạch giao thông.",
        "category": "hoc_tap",
        "keywords": ["vận tải", "kinh tế", "logistics", "học"]
    },
    {
        "question": "ngành công trình học những môn gì",
        "answer": "Ngành Công trình gồm: Cơ học đất, Nền móng, Kết cấu thép, Kết cấu bê tông, Thiết kế cầu, Thiết kế đường, Thí nghiệm vật liệu.",
        "category": "hoc_tap",
        "keywords": ["công trình", "xây dựng", "cầu đường", "học"]
    },
    {
        "question": "học phí một năm bao nhiêu",
        "answer": "Học phí dao động từ 15-25 triệu đồng/năm tùy ngành. Hệ chất lượng cao có học phí khoảng 30-40 triệu đồng/năm.",
        "category": "hoc_tap",
        "keywords": ["học phí", "tiền học", "chi phí", "bao nhiêu"]
    },
    {
        "question": "một tín chỉ bao nhiêu tiền",
        "answer": "Giá một tín chỉ khoảng 450.000 - 550.000 đồng tùy theo ngành học và hệ đào tạo.",
        "category": "hoc_tap",
        "keywords": ["tín chỉ", "học phí", "giá", "bao nhiêu"]
    },
    {
        "question": "sinh viên năm nhất cần lưu ý gì",
        "answer": "Cần làm quen với phương pháp học đại học, tham gia các hoạt động Đoàn - Hội, tìm hiểu kỹ chương trình đào tạo, xây dựng kế hoạch học tập ngay từ đầu.",
        "category": "hoc_tap",
        "keywords": ["năm nhất", "lưu ý", "kinh nghiệm", "sinh viên mới"]
    },
    
    # ==================== SINH HOẠT (Sinh hoat) ====================
    {
        "question": "sinh viên năm nhất có được ở ký túc xá không",
        "answer": "Có, sinh viên năm nhất được ưu tiên xét duyệt ở ký túc xá. Bạn liên hệ phòng Công tác sinh viên tầng 1 nhà A1 để đăng ký nhé.",
        "category": "sinh_hoat",
        "keywords": ["năm nhất", "ký túc xá", "ở", "đăng ký"]
    },
    {
        "question": "kí túc xá có gần trường không",
        "answer": "Ký túc xá nằm trong khuôn viên trường, cách giảng đường chỉ 5-10 phút đi bộ, rất thuận tiện cho việc học tập.",
        "category": "sinh_hoat",
        "keywords": ["ký túc xá", "gần", "vị trí", "cách"]
    },
    {
        "question": "giá phòng ký túc xá bao nhiêu",
        "answer": "Giá phòng ký túc xá từ 300.000 - 600.000 đồng/tháng tùy loại phòng (4 người, 6 người, 8 người).",
        "category": "sinh_hoat",
        "keywords": ["ký túc xá", "giá", "tiền phòng", "bao nhiêu"]
    },
    {
        "question": "căn tin trường có những món gì",
        "answer": "Căn tin trường có đa dạng món ăn: cơm văn phòng, bún phở, mì tôm, trà sữa, đồ ăn nhanh. Giá từ 15.000 - 40.000 đồng/suất.",
        "category": "sinh_hoat",
        "keywords": ["căn tin", "ăn", "đồ ăn", "món"]
    },
    {
        "question": "có chỗ gửi xe cho sinh viên không",
        "answer": "Trường có bãi gửi xe rộng rãi cho sinh viên với giá 3.000 - 5.000 đồng/xe/lượt hoặc đăng ký vé tháng.",
        "category": "sinh_hoat",
        "keywords": ["gửi xe", "bãi xe", "để xe", "xe máy"]
    },
    
    # ==================== VIỆC LÀM (Viec lam) ====================
    {
        "question": "trường có hỗ trợ việc làm sau khi tốt nghiệp không",
        "answer": "Trường có Trung tâm Hỗ trợ sinh viên và kết nối doanh nghiệp để giới thiệu việc làm. Hơn 95% sinh viên có việc làm ngay sau khi ra trường.",
        "category": "viec_lam",
        "keywords": ["việc làm", "tốt nghiệp", "hỗ trợ", "doanh nghiệp"]
    },
    {
        "question": "các công ty thường tuyển sinh viên trường nào",
        "answer": "Sinh viên trường thường được các tập đoàn lớn tuyển dụng: Cienco4, Tổng công ty Cầu đường, Trung Nam Group, Tân Hoàng Cầu, các công ty nước ngoài như JSTI, CRRC...",
        "category": "viec_lam",
        "keywords": ["công ty", "tuyển dụng", "doanh nghiệp", "đối tác"]
    },
    {
        "question": "mức lương khởi điểm của sinh viên sau khi ra trường",
        "answer": "Mức lương khởi điểm dao động từ 7 - 12 triệu đồng/tháng, tùy vào ngành học và năng lực của sinh viên.",
        "category": "viec_lam",
        "keywords": ["lương", "khởi điểm", "thu nhập", "bao nhiêu"]
    },
    {
        "question": "có chương trình thực tập hưởng lương không",
        "answer": "Có, nhiều doanh nghiệp có chương trình thực tập hưởng lương cho sinh viên năm 3 và năm 4. Mức hỗ trợ từ 2 - 5 triệu/tháng.",
        "category": "viec_lam",
        "keywords": ["thực tập", "hưởng lương", "thực tế", "doanh nghiệp"]
    },
    
    # ==================== ĐIỂM SỐ - HỌC BỔNG (Diem so - Hoc bong) ====================
    {
        "question": "điểm rèn luyện để làm gì",
        "answer": "Điểm rèn luyện dùng để xét học bổng, đánh giá danh hiệu sinh viên và là tiêu chí để xét tốt nghiệp. Cố gắng tham gia nhiều hoạt động nhé!",
        "category": "diem_so",
        "keywords": ["điểm rèn luyện", "học bổng", "tốt nghiệp"]
    },
    {
        "question": "học bổng khuyến khích học tập yêu cầu gì",
        "answer": "Bạn cần có điểm trung bình học kỳ đạt loại Khá trở lên (thường từ 2.5/4.0 hoặc 7.0/10) và điểm rèn luyện Khá trở lên, không nợ môn.",
        "category": "hoc_bong",
        "keywords": ["học bổng", "khuyến khích", "điều kiện", "yêu cầu"]
    },
    {
        "question": "học bổng tài trợ có những loại nào",
        "answer": "Các loại học bổng: Học bổng khuyến khích học tập, Học bổng doanh nghiệp (JSTI, Trung Nam, Cienco4), Học bổng tài năng trẻ, Học bổng vượt khó.",
        "category": "hoc_bong",
        "keywords": ["học bổng", "tài trợ", "doanh nghiệp", "loại"]
    },
    {
        "question": "cách tính điểm trung bình tích lũy",
        "answer": "Điểm trung bình tích lũy = Tổng (Điểm môn x Số tín chỉ) / Tổng số tín chỉ đã học. Hệ số 4 tương ứng: A(4.0), B+(3.5), B(3.0), C+(2.5), C(2.0), D+(1.5), D(1.0), F(0)",
        "category": "diem_so",
        "keywords": ["điểm", "trung bình", "tích lũy", "tính"]
    },
    {
        "question": "bao nhiêu điểm là qua môn",
        "answer": "Từ điểm D (1.0/4.0) trở lên là qua môn. Tuy nhiên, để đạt học bổng cần đạt từ điểm C (2.0/4.0) trở lên.",
        "category": "diem_so",
        "keywords": ["qua môn", "điểm", "tối thiểu", "mấy điểm"]
    },
    {
        "question": "học lại có phải đóng tiền không",
        "answer": "Có, học lại phải đóng học phí theo số tín chỉ đăng ký học lại, giá mỗi tín chỉ tương tự như học chính.",
        "category": "diem_so",
        "keywords": ["học lại", "đóng tiền", "học phí", "bao nhiêu"]
    },
    
    # ==================== THỦ TỤC HÀNH CHÍNH (Thu tuc hanh chinh) ====================
    {
        "question": "muốn xin bảng điểm thì lên đâu",
        "answer": "Để xin bảng điểm, bạn lên Phòng Đào tạo tại tầng 1 tòa nhà A1. Nhớ mang theo Thẻ sinh viên nhé.",
        "category": "thu_tuc",
        "keywords": ["bảng điểm", "xin", "phòng đào tạo"]
    },
    {
        "question": "mất thẻ sinh viên thì làm lại ở đâu",
        "answer": "Nếu mất thẻ, bạn hãy báo ngay cho Phòng Công tác sinh viên (Tầng 1 nhà A1) để làm thủ tục cấp lại thẻ mới.",
        "category": "thu_tuc",
        "keywords": ["mất thẻ", "thẻ sinh viên", "làm lại", "cấp lại"]
    },
    {
        "question": "xin giấy xác nhận sinh viên ở đâu",
        "answer": "Bạn lên Phòng Công tác sinh viên (Tầng 1 nhà A1) để xin giấy xác nhận. Nhớ mang theo Thẻ sinh viên và đóng lệ phí 10.000đ.",
        "category": "thu_tuc",
        "keywords": ["giấy xác nhận", "xác nhận sinh viên", "xin giấy"]
    },
    {
        "question": "thủ tục xin phép nghỉ học",
        "answer": "Bạn viết đơn xin phép có xác nhận của phụ huynh và nộp cho Cố vấn học tập. Nếu nghỉ quá 20% số buổi sẽ không được dự thi.",
        "category": "thu_tuc",
        "keywords": ["nghỉ học", "xin phép", "thủ tục", "đơn"]
    },
    {
        "question": "thủ tục bảo lưu kết quả học tập",
        "answer": "Bạn làm đơn xin bảo lưu, được Cố vấn học tập xác nhận, nộp Phòng Đào tạo. Thời gian bảo lưu tối đa 2 năm.",
        "category": "thu_tuc",
        "keywords": ["bảo lưu", "kết quả", "tạm dừng", "thủ tục"]
    },
    {
        "question": "thủ tục chuyển ngành học",
        "answer": "Sinh viên có thể xin chuyển ngành sau học kỳ đầu tiên nếu đáp ứng điều kiện về điểm số và có nguyện vọng phù hợp.",
        "category": "thu_tuc",
        "keywords": ["chuyển ngành", "đổi ngành", "thủ tục", "điều kiện"]
    },
    {
        "question": "cách đăng ký môn học online",
        "answer": "Đăng nhập vào hệ thống tín chỉ tại http://daotao.utc.edu.vn, chọn học kỳ, chọn môn học và xác nhận đăng ký trước hạn.",
        "category": "thu_tuc",
        "keywords": ["đăng ký môn", "online", "tín chỉ", "cách"]
    },
    
    # ==================== LỊCH HỌC - THI CỬ (Lich hoc - Thi cu) ====================
    {
        "question": "lịch nghỉ tết năm nay thế nào",
        "answer": "Lịch nghỉ Tết chính thức sẽ được Phòng Đào tạo thông báo trên website trước Tết khoảng 3-4 tuần. Thường sinh viên được nghỉ 2-3 tuần.",
        "category": "lich_hoc",
        "keywords": ["nghỉ tết", "lịch", "tết", "nghỉ"]
    },
    {
        "question": "đăng ký học lại như thế nào",
        "answer": "Sinh viên đăng ký học lại trực tuyến trên trang tín chỉ của trường vào các đợt đăng ký môn học đầu học kỳ, hoặc cuối học kỳ trước.",
        "category": "hoc_tap",
        "keywords": ["học lại", "đăng ký", "tín chỉ"]
    },
    {
        "question": "thi lại khi nào",
        "answer": "Lịch thi lại được tổ chức sau khi có kết quả thi chính thức khoảng 2-3 tuần. Theo dõi thông báo từ Phòng Đào tạo hoặc website trường.",
        "category": "thi_cu",
        "keywords": ["thi lại", "lịch thi", "khi nào"]
    },
    {
        "question": "khi nào có lịch thi cuối kỳ",
        "answer": "Lịch thi cuối kỳ thường được Phòng Đào tạo công bố vào giữa học kỳ, khoảng tuần thứ 8-10 của học kỳ.",
        "category": "thi_cu",
        "keywords": ["lịch thi", "cuối kỳ", "khi nào"]
    },
    {
        "question": "thi vấn đáp là gì",
        "answer": "Thi vấn đáp là hình thức thi trực tiếp với giảng viên, sinh viên trả lời câu hỏi và bảo vệ kiến thức, thường áp dụng cho các môn chuyên ngành.",
        "category": "thi_cu",
        "keywords": ["thi vấn đáp", "hình thức thi", "trực tiếp"]
    },
    {
        "question": "điều kiện dự thi cuối kỳ",
        "answer": "Sinh viên phải có điểm chuyên cần, đủ số buổi học theo quy định (tối thiểu 80% số buổi) và hoàn thành bài tập, đồ án môn học.",
        "category": "thi_cu",
        "keywords": ["dự thi", "điều kiện", "cuối kỳ"]
    },
    
    # ==================== HOẠT ĐỘNG - CÂU LẠC BỘ (Hoat dong - CLB) ====================
    {
        "question": "trường có câu lạc bộ nào hay không",
        "answer": "Trường có rất nhiều CLB đa dạng: CLB Tiếng Anh, CLB Âm nhạc, CLB Robotic, CLB Truyền thông, CLB Tình nguyện, CLB Bóng đá, CLB Bóng rổ. Theo dõi Fanpage Đoàn Thanh niên để đăng ký.",
        "category": "hoat_dong",
        "keywords": ["câu lạc bộ", "clb", "hoạt động", "tham gia"]
    },
    {
        "question": "cách tham gia nghiên cứu khoa học",
        "answer": "Liên hệ với giảng viên hướng dẫn hoặc Phòng Khoa học Công nghệ. Trường có nhiều đề tài NCKH cho sinh viên tham gia hàng năm. Có hỗ trợ kinh phí lên đến 10 triệu đồng/đề tài.",
        "category": "nghien_cuu",
        "keywords": ["nghiên cứu khoa học", "nckh", "tham gia", "đề tài"]
    },
    {
        "question": "có thể tham gia nhiều CLB cùng lúc không",
        "answer": "Có, sinh viên có thể tham gia nhiều CLB cùng lúc tùy theo quỹ thời gian và sở thích cá nhân. Mỗi CLB đều có lịch sinh hoạt riêng.",
        "category": "hoat_dong",
        "keywords": ["clb", "tham gia", "nhiều", "cùng lúc"]
    },
    {
        "question": "các hoạt động tình nguyện của trường",
        "answer": "Trường tổ chức các chiến dịch tình nguyện: Mùa hè xanh, Hiến máu nhân đạo, Xuân tình nguyện, Tiếp sức mùa thi. Rất bổ ích cho sinh viên.",
        "category": "hoat_dong",
        "keywords": ["tình nguyện", "hiến máu", "mùa hè xanh", "hoạt động"]
    },
    {
        "question": "các giải thể thao trong trường",
        "answer": "Hàng năm có các giải: Bóng đá truyền thống, Bóng rổ sinh viên, Cầu lông, Bóng bàn, Điền kinh. Đăng ký qua Đoàn Thanh niên hoặc Khoa thể thao.",
        "category": "hoat_dong",
        "keywords": ["thể thao", "giải đấu", "bóng đá", "bóng rổ"]
    },
    
    # ==================== TIỆN ÍCH - CƠ SỞ VẬT CHẤT (Tien ich) ====================
    {
        "question": "trường có wifi không",
        "answer": "Có, toàn bộ khuôn viên trường được phủ wifi miễn phí. Tên wifi: UTC-WIFI, mật khẩu: utc@2024. Có vùng phủ sóng mạnh tại các tòa nhà A1, A2, A4, thư viện.",
        "category": "tien_ich",
        "keywords": ["wifi", "internet", "mạng", "kết nối"]
    },
    {
        "question": "thư viện mở cửa đến mấy giờ",
        "answer": "Thư viện mở cửa từ 7h30 đến 21h00 các ngày trong tuần. Cuối tuần mở từ 8h00 đến 17h00. Thư viện có phòng đọc máy tính và phòng đọc sách riêng.",
        "category": "tien_ich",
        "keywords": ["thư viện", "giờ mở", "sách", "mượn sách"]
    },
    {
        "question": "có thể mượn sách về nhà không",
        "answer": "Có, bạn có thể mượn sách về nhà với thời hạn 2 tuần/lần. Thủ tục: mang thẻ sinh viên lên quầy thư viện đăng ký mượn.",
        "category": "tien_ich",
        "keywords": ["mượn sách", "thư viện", "về nhà"]
    },
    {
        "question": "có máy tính cho sinh viên thực hành không",
        "answer": "Có, trường có các phòng máy thực hành tại tầng 5 tòa A4 và tầng 2 tòa A7. Sinh viên đăng ký qua website để sử dụng miễn phí.",
        "category": "tien_ich",
        "keywords": ["máy tính", "thực hành", "phòng máy", "lab"]
    },
    {
        "question": "có chỗ nào để in ấn tài liệu không",
        "answer": "Có, có nhiều tiệm photo trong khuôn viên trường: gần căn tin, tầng 1 nhà A1, cạnh thư viện. Giá in 200-500đ/trang tùy loại.",
        "category": "tien_ich",
        "keywords": ["in ấn", "photo", "tài liệu", "tiệm in"]
    },
    {
        "question": "có bãi đỗ xe ô tô không",
        "answer": "Có, trường có bãi đỗ xe ô tô dành cho giảng viên và sinh viên. Phí gửi xe 10.000đ/lượt hoặc 200.000đ/tháng.",
        "category": "tien_ich",
        "keywords": ["bãi xe", "ô tô", "đỗ xe", "gửi xe ô tô"]
    },
    {
        "question": "có phòng y tế không",
        "answer": "Có, phòng y tế nằm ở tầng 1 nhà B3, gần cổng chính. Phục vụ khám chữa bệnh thông thường và cấp thuốc cơ bản cho sinh viên.",
        "category": "tien_ich",
        "keywords": ["phòng y tế", "bác sĩ", "khám bệnh", "y tế"]
    },
    
]

# ============================================================
# TẠO CÁC BIẾN THỂ CÂU HỎI NÂNG CAO
# ============================================================

def generate_advanced_question_variations(original_question: str, answer: str, keywords: List[str]) -> List[Dict]:
    variations = []
    
    question_templates = [
        "{}", "cho tôi hỏi {}", "bạn ơi {}", "xin hỏi {}", "robot ơi {}", 
        "tôi muốn biết {}", "hãy cho tôi biết {}", "{} ạ", "{} không", "{} vậy"
    ]
    
    replacement_maps = {
        "ở đâu": ["nằm ở đâu", "chỗ nào", "tại đâu", "địa chỉ"],
        "khi nào": ["lúc nào", "thời gian nào", "bao giờ", "ngày nào"],
        "bao nhiêu": ["mấy", "những", "số lượng", "chi phí", "giá cả"],
        "thế nào": ["ra sao", "như thế nào", "cách nào", "làm sao"]
    }
    
    for template in question_templates:
        if "{}" in template:
            var_text = template.format(original_question)
            variations.append({
                "question": var_text,
                "answer": answer,
                "keywords": keywords
            })
            
    for old_word, new_words in replacement_maps.items():
        if old_word in original_question:
            for new_word in new_words:
                var_text = original_question.replace(old_word, new_word)
                variations.append({
                    "question": var_text,
                    "answer": answer,
                    "keywords": keywords
                })
                
    typo_variations = [
        ("trường", "truong"), ("khoa", "khoa"), ("phòng", "phong"),
        ("năm", "nam"), ("học", "hoc"), ("sinh viên", "sinh vien"),
    ]
    
    for correct, typo in typo_variations:
        if correct in original_question:
            var_text = original_question.replace(correct, typo)
            if var_text != original_question:
                variations.append({
                    "question": var_text,
                    "answer": answer,
                    "keywords": keywords
                })
                
    return variations


def build_full_qa_dataset() -> List[Dict]:
    full_dataset = []
    seen_questions = set()
    
    for item in QA_DATA:
        # Add original
        q_norm = item["question"].lower().strip()
        if q_norm not in seen_questions:
            seen_questions.add(q_norm)
            full_dataset.append(item)
            
        # Add variations
        variations = generate_advanced_question_variations(
            item["question"], item["answer"], item["keywords"]
        )
        
        for var in variations:
            q_var_norm = var["question"].lower().strip()
            if q_var_norm not in seen_questions:
                seen_questions.add(q_var_norm)
                var["category"] = item.get("category", "khac")
                full_dataset.append(var)
                
    return full_dataset


def generate_custom_qa():
    output_path = os.path.join(os.path.dirname(__file__), "data", "qa_custom.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    full_dataset = build_full_qa_dataset()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"✅ TẠO FILE THÀNH CÔNG!")
    print(f"Tổng số cặp QA: {len(full_dataset)}")
    print(f"File lưu tại: {output_path}")

if __name__ == "__main__":
    generate_custom_qa()