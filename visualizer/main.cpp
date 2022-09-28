#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

class Paint : public olc::PixelGameEngine
{
public:
	Paint()
	{
		sAppName = "digit drawer";
	}

private:
    std::vector<std::array<float, 2>> pixels;
    float x = 50.0f;
    float y = 50.0f;

public:
	bool OnUserCreate() override
	{
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
        // Erase previous frame
	    Clear(olc::BLACK);

        // Draw
        for (int i = 0; i < pixels.size(); i++)
        {
            //FillRect(20, ScreenHeight() - 20, 40, 10, olc::GREEN);
            FillRect(int(pixels[i][0]), int(pixels[i][1]), 1, 1, olc::WHITE);
        }

        // Cursor
        FillCircle(int(x), int(y), 1, olc::WHITE);

        x = float(GetMouseX());
        y = float(GetMouseY());
        
        // Paint
        if (GetMouse(0).bHeld)
            pixels.push_back({float(GetMouseX()), float(GetMouseY())});

		return true;
	}
};

int main()
{
	Paint digit_drawer;
	if (digit_drawer.Construct(300, 200, 2, 2))
		digit_drawer.Start();
	return 0;
}
