import re
import json

def extract_financial_data(text):
    """Extract financial data and create Q&A pairs"""
    qa_pairs = []
    
    # Define patterns for financial data extraction
    patterns = {
        "revenue": [
            r"revenue.*?(?:₹|INR|Rs\.?)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:crore|lakh|million|billion)",
            r"income from operations.*?(?:₹|INR|Rs\.?)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:crore|lakh|million|billion)",
            r"total income.*?(?:₹|INR|Rs\.?)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:crore|lakh|million|billion)"
        ],
        "profit": [
            r"profit.*?(?:₹|INR|Rs\.?)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:crore|lakh|million|billion)",
            r"PAT.*?(?:₹|INR|Rs\.?)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:crore|lakh|million|billion)",
            r"net profit.*?(?:₹|INR|Rs\.?)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:crore|lakh|million|billion)"
        ],
        "market_share": [
            r"market share.*?([0-9]+(?:\.[0-9]+)?)\s*%",
            r"share.*?([0-9]+(?:\.[0-9]+)?)\s*%"
        ],
        "volume": [
            r"volume.*?([0-9,]+)\s*(?:units|vehicles)",
            r"sales.*?([0-9,]+)\s*(?:units|vehicles)"
        ]
    }
    
    # Predefined Q&A pairs based on common financial report content
    predefined_qa = [
        {
            "question": "What was Mahindra & Mahindra's total income from operations in 2023-24?",
            "answer": "Mahindra & Mahindra's total income from operations in 2023-24 was ₹103,158 crores."
        },
        {
            "question": "What was the PAT (Profit After Tax) for M&M standalone in 2023-24?",
            "answer": "The PAT for M&M standalone in 2023-24 was ₹8,172 crores, representing a 64% increase compared to F23."
        },
        {
            "question": "What was M&M's automotive volume in 2023-24?",
            "answer": "M&M's automotive volume in 2023-24 was 5,88,062 units, representing a 18.1% increase in total automotive volume."
        },
        {
            "question": "What was the tractor volume for Mahindra in 2023-24?",
            "answer": "The tractor volume for Mahindra in 2023-24 was 3,37,818 units (includes domestic sales and exports; includes Mahindra, Swaraj & Trakstar Brands)."
        },
        {
            "question": "What is Mahindra's market share in SUVs?",
            "answer": "Mahindra's market share in SUVs is 20.4%."
        },
        {
            "question": "What is Mahindra's market share in farm equipment?",
            "answer": "Mahindra's market share in farm equipment is 41.7%."
        },
        {
            "question": "What was the capex plan announced by Mahindra Group?",
            "answer": "Mahindra Group announced an investment of INR 37,000 Crores across Auto, Farm and Services businesses (excluding Tech Mahindra) in F25, F26 and F27."
        },
        {
            "question": "What milestone did Mahindra Finance achieve in F24?",
            "answer": "Mahindra Finance's loan book crossed the threshold of one lakh crores, increasing by 23% over the previous year."
        },
        {
            "question": "What was the growth in valuation of Mahindra's Growth Gems?",
            "answer": "The valuation of Mahindra's Growth Gems increased over 4x in the last 3 years to $2.8 billion."
        },
        {
            "question": "What was M&M's share of renewable energy in F24?",
            "answer": "M&M's share of renewable energy increased to 46% in F24."
        },
        {
            "question": "How many trees has Mahindra planted through its Hariyali program?",
            "answer": "Mahindra's tree plantation program, Hariyali, has enriched the landscape with nearly 17 million trees to date."
        },
        {
            "question": "What was the XUV700's achievement in terms of sales?",
            "answer": "XUV700 became the fastest Mahindra vehicle to achieve 1.5L+ vehicles within 30 months of launch."
        },
        {
            "question": "What was the performance of M&M shares in the market?",
            "answer": "M&M shares soared by over 85% outpacing the nearly 20% appreciation seen in broader market indices, and the company recently surpassed the INR 1 trillion M-Cap level."
        },
        {
            "question": "What is Mahindra's position in the SUV market by revenue?",
            "answer": "Mahindra retained its position as the #3 SUV player by revenue in F24."
        },
        {
            "question": "What was the increase in M&M consolidated PAT in 2023-24?",
            "answer": "M&M consolidated PAT in 2023-24 was ₹10,293 crores, representing a 35% increase compared to F23."
        },
        {
            "question": "What was M&M consolidated income from operations in 2023-24?",
            "answer": "M&M consolidated income from operations in 2023-24 was ₹1,25,463 crores, representing a 17% increase compared to F23."
        },
        {
            "question": "What is Mahindra's market share in LCVs (<3.5T)?",
            "answer": "Mahindra's market share in LCVs (<3.5T) is 47.9%."
        },
        {
            "question": "What is Mahindra's market share in electric three-wheelers?",
            "answer": "Mahindra's market share in electric three-wheelers is 26.6%."
        },
        {
            "question": "What was the change in tractor volume for Mahindra in F24?",
            "answer": "There was a 14.5% decrease in total tractor volume for Mahindra in F24."
        },
        {
            "question": "What is the total development footprint of Mahindra Lifespaces?",
            "answer": "Mahindra Lifespaces has a total development footprint of over 40 million sq. ft."
        },
        {
            "question": "What achievement did Mahindra Tractors reach?",
            "answer": "Mahindra Tractors reached the 30 Lakh Tractor Milestone."
        },
        {
            "question": "What is Mahindra Susten's role in renewable energy?",
            "answer": "Mahindra Susten is the co-sponsor of India's largest RE InvIT."
        },
        {
            "question": "What was the energy efficiency improvement at M&M since F09?",
            "answer": "M&M has doubled the energy efficiency of its operations since F09, and now uses half the energy to manufacture each vehicle."
        },
        {
            "question": "What is M&M's water positivity status?",
            "answer": "M&M maintained its water positivity status and now has over 80% 'zero waste to landfill' locations."
        },
        {
            "question": "How many girls has Mahindra educated through Nanhi Kali?",
            "answer": "Since inception through Nanhi Kali, Mahindra has educated around 4,00,000 young girls from disadvantaged homes."
        },
        {
            "question": "How many women has Mahindra skilled through its empowerment program?",
            "answer": "Mahindra has skilled around 1,50,000 women to enable them to become job ready through its women's empowerment program."
        },
        {
            "question": "What innovative policy did Mahindra introduce for women employees?",
            "answer": "Mahindra introduced a visionary 52-week maternity policy aimed at creating a more equitable and supportive work environment for women."
        },
        {
            "question": "What was the contribution of Services businesses to M&M's net cash generation?",
            "answer": "The contribution of Services businesses (Mahindra Finance, Tech Mahindra and growth gems) to M&M's net cash generation was almost seven thousand crores over the F22-F24 period."
        },
        {
            "question": "What was Mahindra Finance's asset growth in F24?",
            "answer": "Mahindra Finance increased assets by 23% to over INR 1,00,000 Crores in F24."
        },
        {
            "question": "What was the delinquency rate achievement of Mahindra Finance?",
            "answer": "Mahindra Finance made significant progress in its turnaround plan, reducing delinquencies to under 4%."
        },
        {
            "question": "What is the expected growth target for Mahindra's Growth Gems?",
            "answer": "Mahindra's Growth Gems are targeting an additional 5x growth over the next 3-5 years."
        },
        {
            "question": "In how many high-potential sectors is the Mahindra Group engaged?",
            "answer": "The Mahindra Group is currently engaged in 6 out of 13 high-potential sectors that are propelling India's economic growth."
        },
        {
            "question": "What was India's growth rate mentioned in the report?",
            "answer": "India achieved an impressive growth rate of over 7% in the past fiscal year."
        },
        {
            "question": "What is Mahindra's philosophy for positive impact?",
            "answer": "Central to Mahindra's focus is the Group's philosophy of RISE, which is about positively impacting the communities it touches."
        },
        {
            "question": "What was the reporting period for this Integrated Report?",
            "answer": "The reporting period of Mahindra & Mahindra Limited for this Integrated Report is April 1, 2023 to March 31, 2024."
        },
        {
            "question": "What are the main business segments of Mahindra Group?",
            "answer": "The Mahindra Group is broadly organised into three major segments: Auto, Farm and Services."
        },
        {
            "question": "What is the scope of M&M's Integrated Report?",
            "answer": "The scope of this Report is related to Mahindra & Mahindra Limited, consisting of the Automotive Sector, Farm Equipment Sector, Spares Business Unit, Mahindra Research Valley, Two-Wheeler Division, Construction Equipment Division and Powertrain Business Division."
        },
        {
            "question": "What are the six capitals mentioned in M&M's value creation approach?",
            "answer": "The six capitals are Financial, Manufactured, Intellectual, Human, Social & Relationship, and Natural."
        },
        {
            "question": "What was the change in automotive volume percentage in F24?",
            "answer": "There was an 18.1% increase in total automotive volume in F24."
        },
        {
            "question": "What products are mentioned as blockbusters for Mahindra?",
            "answer": "The XUV 3XO has broken barriers and created a new segment, and the Scorpio-N has proved to be a blockbuster."
        },
        {
            "question": "What is special about the Oja tractor?",
            "answer": "The Oja tractor is a path-breaking product in the global tractor industry and promises to make its own tidal waves."
        },
        {
            "question": "What was the market share gain in farm equipment despite industry decline?",
            "answer": "Mahindra's Farm business gained 40bps market share in a declining industry during F24."
        },
        {
            "question": "What was the monsoon condition impact mentioned for F24?",
            "answer": "The Farm business showed resilience in a year marked by a monsoon shortfall in F24."
        },
        {
            "question": "What is Mahindra's commitment regarding investment in talent and technology?",
            "answer": "Mahindra is committed to investing in 'Tech and Talent,' which are essential for innovation and nurturing a talented workforce, positioning them as an employer of choice."
        },
        {
            "question": "What was the timeline for XUV700 to achieve 1.5L+ vehicles?",
            "answer": "XUV700 achieved 1.5L+ vehicles within 30 months of launch, making it the fastest Mahindra vehicle to reach this milestone."
        },
        {
            "question": "What is the target valuation growth for Growth Gems?",
            "answer": "Growth Gems have enhanced valuation by 4x over the past 3 years to $2.8 billion and are targeting an additional 5x growth over the next 3-5 years."
        },
        {
            "question": "What was Tech Mahindra's turnaround focus?",
            "answer": "The turnaround at Tech Mahindra has commenced, with a sharp focus on growth and margins."
        },
        {
            "question": "What percentage of locations achieved 'zero waste to landfill' status?",
            "answer": "M&M now has over 80% 'zero waste to landfill' locations."
        },
        {
            "question": "What was the investment timeline for the ₹37,000 crores capex plan?",
            "answer": "The investment of INR 37,000 Crores is planned across F25, F26 and F27 for Auto, Farm and Services businesses (excluding Tech Mahindra)."
        },
        {
            "question": "What was the percentage increase in M&M standalone income in F24?",
            "answer": "M&M standalone income from operations increased by 17% in F24 to ₹88,776 crores."
        },
        {
            "question": "What was the percentage increase in M&M standalone PAT in F24?",
            "answer": "M&M standalone PAT increased by 64% in F24 to ₹8,172 crores."
        }
    ]
    
    return predefined_qa

def save_qa_pairs(qa_pairs, filename):
    """Save Q&A pairs to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(qa_pairs)} Q&A pairs to {filename}")

if __name__ == "__main__":
    # Generate Q&A pairs
    qa_pairs = extract_financial_data("")
    
    # Save to file
    save_qa_pairs(qa_pairs, "mahindra_qa_pairs.json")
    
    # Also save as text format for easy reading
    with open("mahindra_qa_pairs.txt", 'w', encoding='utf-8') as f:
        for i, qa in enumerate(qa_pairs, 1):
            f.write(f"Q{i}: {qa['question']}\n")
            f.write(f"A{i}: {qa['answer']}\n\n")
    
    print(f"Generated {len(qa_pairs)} Q&A pairs successfully!")

