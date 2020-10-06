import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
baseurl = os.path.dirname(os.path.abspath(__file__))
from util.file_helper import FileReader


class MemberModel:

    def __init__(self):
        self.filereader = FileReader()

    def hook_process(self):
        members = self.get_memberList()

    def get_memberList(self):
        reader = self.filereader
        reader.context = os.path.join(baseurl, 'data')
        reader.fname = 'member_detail.csv'
        members = reader.csv_to_dframe()
        print(f'memberList:\n{members.head()}')
        return members


if __name__ == '__main__':
    member = MemberModel()
    member.hook_process()
